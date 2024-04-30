use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::iter;
use std::ops::Range;
use std::rc::Rc;

use wasm::FuncType;
use wasm_encoder as wasm;
use indexmap::IndexSet;

use crate::ir1::{DebruijnIndex, Op, Value, Global};
use crate::ir2::{GlobalDef, Expr};
use crate::runtime::Runtime;

pub fn translate<'a>(global_defs: &[GlobalDef<'a>], main: usize) -> Vec<u8> {
    // TODO: configure this or include it at compile time
    let mut runtime_bytes = Vec::new();
    std::fs::File::open("target/wasm32-unknown-unknown/release/clocky_runtime.wasm").unwrap()
        .read_to_end(&mut runtime_bytes).unwrap();
    let runtime = Runtime::from_bytes(&runtime_bytes);

    let mut codes = wasm::CodeSection::new();
    let mut data = wasm::DataSection::new();
    let mut functions = wasm::FunctionSection::new();
    let mut exports = wasm::ExportSection::new();
    // exports.export("memory", wasm::ExportKind::Memory, 0);
    let mut function_types = FunctionTypes { types: runtime.types.clone() };
    let mut globals_out = wasm::GlobalSection::new();
    let mut init_func = wasm::Function::new_with_locals_types([wasm::ValType::I32]);

    runtime.emit_functions(&mut functions);
    runtime.emit_globals(&mut globals_out);

    let heap_global = globals_out.len();
    globals_out.global(
        wasm::GlobalType {
            val_type: wasm::ValType::I32,
            mutable: true,
            shared: false,
        },
        &wasm::ConstExpr::i32_const(0)
    );
    let bad_global = globals_out.len();
    globals_out.global(
        wasm::GlobalType {
            val_type: wasm::ValType::I32,
            mutable: true,
            shared: false,
        },
        &wasm::ConstExpr::i32_const(0)
    );
    let globals_offset = globals_out.len();
    exports.export("bad", wasm::ExportKind::Global, bad_global);

    runtime.emit_code(&mut codes);
    runtime.emit_data(&mut data);

    let func_offset = runtime.functions.len() as u32;
    let mut translator = Translator {
        globals: global_defs,
        globals_offset,
        func_table_offset: func_offset, // TODO: is this right?
        function_types: &mut function_types,
        heap_global,
        bad_global,
        runtime_exports: &runtime.exports,
    };

    for (i, def) in global_defs.into_iter().enumerate() {
        println!("doing {i}");
        globals_out.global(wasm::GlobalType {
            val_type: wasm::ValType::I32,
            mutable: true,
            shared: false,
        }, &wasm::ConstExpr::i32_const(0));
        match def {
            GlobalDef::Func { rec, arity, env_size, body } => {
                {
                let mut trans = FuncTranslator::new(&mut translator, *rec, *env_size, *arity);
                let ctx = trans.make_initial_ctx();
                trans.translate(ctx, body);
                if i == 12 {
                    println!("func 12 locals: {:?}", trans.locals);
                    println!("func 12 code: {:?}", trans.insns);
                }
                let (type_idx, func) = trans.finish();
                functions.function(type_idx);
                codes.function(&func);

                init_func.instruction(&wasm::Instruction::GlobalGet(heap_global));
                init_func.instruction(&wasm::Instruction::LocalTee(0));
                init_func.instruction(&wasm::Instruction::I32Const(func_offset as i32 + i as i32));
                init_func.instruction(&wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::I32Const(*arity as i32));
                init_func.instruction(&wasm::Instruction::I32Store(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::I32Const(8));
                init_func.instruction(&wasm::Instruction::I32Add);
                init_func.instruction(&wasm::Instruction::GlobalSet(heap_global));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::GlobalSet(globals_offset + i as u32));
                }
            },
            GlobalDef::ClosedExpr { body } => {
                let mut trans = FuncTranslator::new(&mut translator, false, 0, 0);
                let ctx = trans.make_initial_ctx();
                trans.translate(ctx, body);
                let (type_idx, func) = trans.finish();
                functions.function(type_idx);
                codes.function(&func);

                init_func.instruction(&wasm::Instruction::I32Const(0));
                init_func.instruction(&wasm::Instruction::Call(func_offset + i as u32));
                init_func.instruction(&wasm::Instruction::GlobalSet(globals_offset + i as u32));
            },
        }
    }

    println!("main is {main}");
    exports.export("main", wasm::ExportKind::Global, globals_offset + main as u32);
    runtime.emit_exports(&mut exports);

    init_func.instruction(&wasm::Instruction::End);
    let init_func_type = function_types.types.insert_full(
        FuncType::new(iter::empty(), iter::empty())
    ).0 as u32;
    exports.export("init", wasm::ExportKind::Func, functions.len());
    functions.function(init_func_type);
    codes.function(&init_func);

    let mut types = wasm::TypeSection::new();
    for func_type in function_types.types.into_iter() {
        types.function(func_type.params().iter().copied(), func_type.results().iter().copied());
    }
    types.function(vec![wasm::ValType::I32], vec![wasm::ValType::F32]);
    types.function(vec![], vec![]);

    let mut memories = wasm::MemorySection::new();
    memories.memory(wasm::MemoryType {
        minimum: 1 << 15,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut tables = wasm::TableSection::new();
    tables.table(wasm::TableType {
        element_type: wasm::RefType::FUNCREF,
        minimum: func_offset + global_defs.len() as u32,
        maximum: Some(func_offset + global_defs.len() as u32),
    });

    let mut elems = wasm::ElementSection::new();
    // TODO: initialize runtime's table too?
    let function_idxs: Vec<u32> = (func_offset..func_offset + global_defs.len() as u32).collect();
    let function_elems = wasm::Elements::Functions(&function_idxs);
    elems.active(Some(0), &wasm::ConstExpr::i32_const(func_offset as i32), function_elems);

    let mut module = wasm::Module::new();
    module.section(&types);
    module.section(&functions);
    module.section(&tables);
    module.section(&memories);
    module.section(&globals_out);
    module.section(&exports);
    module.section(&wasm::StartSection { function_index: functions.len() - 1 });
    module.section(&elems);
    module.section(&codes);

    module.finish()
}

enum Ctx {
    Empty,
    // TODO: generalize to support multiple locals per variable, to allow for unboxed layouts and such
    Local(u32, Rc<Ctx>),
}

impl Ctx {
    fn lookup(&self, i: DebruijnIndex) -> u32 {
        match (self, i) {
            (&Ctx::Local(n, _), DebruijnIndex(0)) => n,
            (&Ctx::Local(_, ref next), DebruijnIndex(j)) => next.lookup(DebruijnIndex(j-1)),
            (_, _) => panic!("lookup index out of bounds??"),
        }
    }
}

struct FunctionTypes {
    types: IndexSet<FuncType>,
}

impl FunctionTypes {
    fn for_args(&mut self, n: u32) -> u32 {
        let ty = FuncType::new(iter::repeat(wasm::ValType::I32).take(n as usize),
                               iter::once(wasm::ValType::I32));
        self.types.insert_full(ty).0 as u32
    }
}

struct Translator<'a> {
    globals: &'a [GlobalDef<'a>],
    globals_offset: u32,
    func_table_offset: u32,
    function_types: &'a mut FunctionTypes,
    heap_global: u32,
    bad_global: u32,
    runtime_exports: &'a HashMap<String, (wasmparser::ExternalKind, u32)>,
}

struct FuncTranslator<'a, 'b> {
    translator: &'a mut Translator<'b>,
    env_size: u32,
    arity: u32,
    locals: Vec<wasm::ValType>,
    insns: Vec<wasm::Instruction<'a>>,
    rec: bool,
    temps: HashMap<(wasm::ValType, u32), u32>,
}

// closure layout: | code addr | arity | used[0] | used[1] | ... | used[n] |

impl<'a, 'b> FuncTranslator<'a, 'b> {
    fn new(translator: &'a mut Translator<'b>, rec: bool, env_size: u32, arity: u32) -> FuncTranslator<'a, 'b> {
        FuncTranslator {
            translator,
            env_size,
            rec,
            arity,
            locals: iter::repeat(wasm::ValType::I32).take((env_size+arity+1) as usize).collect(),
            insns: Vec::new(),
            temps: HashMap::new(),
        }
    }

    fn args_offset(&self) -> u32 {
        // may in the future be different if the args are unboxed
        self.arity + 1
    }

    fn make_initial_ctx(&mut self) -> Rc<Ctx> {
        // ir2 ctx: environment..., self (if recursive), args...
        // wasm calling convention: args..., closure

        // first load the environment into locals
        let mut ctx = Ctx::Empty.into();
        let closure_local_index = self.args_offset() - 1;
        for i in (0..self.env_size).rev() {
            let env_i_local = self.args_offset() + i;
            ctx = Ctx::Local(env_i_local, ctx).into();
            self.insns.push(wasm::Instruction::LocalGet(closure_local_index));
            // self.insns.push(wasm::Instruction::I32Const(4 * (i as i32 + 1)));
            // self.insns.push(wasm::Instruction::I32Add);
            self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 4 * (i as u64 + 2), align: 2, memory_index: 0 }));
            self.insns.push(wasm::Instruction::LocalSet(env_i_local));
        }

        if self.rec {
            ctx = Ctx::Local(closure_local_index, ctx).into();
        }

        for i in 0..self.arity {
            ctx = Ctx::Local(i, ctx).into();
        }

        ctx
    }

    fn next_local(&mut self, ty: wasm::ValType) -> u32 {
        let n = self.locals.len();
        self.locals.push(ty);
        n as u32
    }

    fn temp(&mut self, ty: wasm::ValType, i: u32) -> u32 {
        *self.temps.entry((ty, i)).or_insert_with(|| {
            let n = self.locals.len();
            self.locals.push(ty);
            n as u32
        })
    }

    fn translate(&mut self, ctx: Rc<Ctx>, expr: &'a Expr<'a>) {
        match *expr {
            Expr::Var(i) => {
                self.insns.push(wasm::Instruction::LocalGet(ctx.lookup(i)));
            },
            Expr::If(e0, e1, e2) => {
                self.translate(ctx.clone(), e0);
                self.insns.push(wasm::Instruction::If(wasm::BlockType::Result(wasm::ValType::I32)));
                self.translate(ctx.clone(), e1);
                self.insns.push(wasm::Instruction::Else);
                self.translate(ctx, e2);
                self.insns.push(wasm::Instruction::End);
            },
            Expr::Let(es, ec) => {
                let mut new_ctx = ctx.clone();
                for &e in es.iter() {
                    // these let bindings do not get access to each other, so use ctx, not new_ctx, here
                    self.translate(ctx.clone(), e);
                    let l = self.next_local(wasm::ValType::I32);
                    self.insns.push(wasm::Instruction::LocalSet(l));
                    // TODO: is this the right order?
                    new_ctx = Ctx::Local(l, new_ctx).into();
                }
                self.translate(new_ctx, ec);
            },
            Expr::Op(op, args) =>
                self.translate_op(ctx, op, args),
            Expr::CallDirect(_, _) =>
                panic!("CallDirect not implemented yet :^)"),
            Expr::CallIndirect(target, args) => {
                for &arg in args.iter() {
                    self.translate(ctx.clone(), arg);
                }
                self.translate(ctx, target);
                let t0 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalTee(t0));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                let t1 = self.temp(wasm::ValType::I32, 1);
                self.insns.push(wasm::Instruction::LocalSet(t1));

                self.insns.push(wasm::Instruction::LocalGet(t1));
                self.insns.push(wasm::Instruction::I32Const(args.len() as i32));
                self.insns.push(wasm::Instruction::I32Eq);
                let if_idx = self.translator.function_types.for_args(args.len() as u32);
                self.insns.push(wasm::Instruction::If(wasm::BlockType::FunctionType(if_idx)));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let funty_idx = self.translator.function_types.for_args(args.len() as u32 + 1);
                self.insns.push(wasm::Instruction::CallIndirect { ty: funty_idx, table: 0 });
                self.insns.push(wasm::Instruction::Else);
                // TODO: create closure!
                self.insns.push(wasm::Instruction::I32Const(1));
                self.insns.push(wasm::Instruction::GlobalSet(self.translator.bad_global));
                for _ in 0..args.len() {
                    self.insns.push(wasm::Instruction::Drop);
                }
                self.insns.push(wasm::Instruction::I32Const(0));
                self.insns.push(wasm::Instruction::End);
            },
        }
    }

    fn dup(&mut self, ty: wasm::ValType) {
        let t = self.temp(ty, 0);
        self.insns.push(wasm::Instruction::LocalTee(t));
        self.insns.push(wasm::Instruction::LocalGet(t));
    }

    fn alloc(&mut self) {
        // TODO: check if we're out of memory lol
        self.insns.push(wasm::Instruction::GlobalGet(self.translator.heap_global));
        let t = self.temp(wasm::ValType::I32, 0);
        self.insns.push(wasm::Instruction::LocalTee(t));
        self.insns.push(wasm::Instruction::I32Add);
        self.insns.push(wasm::Instruction::GlobalSet(self.translator.heap_global));
        self.insns.push(wasm::Instruction::LocalGet(t));
    }

    fn translate_op(&mut self, ctx: Rc<Ctx>, op: Op, args: &'a [&'a Expr<'a>]) {
        match (op, args) {
            (Op::Const(Value::Unit), &[]) => {
                self.insns.push(wasm::Instruction::I32Const(0));
            },
            (Op::Const(Value::Index(i)), &[]) => {
                // eVeRyThInG is boxed
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                self.dup(wasm::ValType::I32);
                self.insns.push(wasm::Instruction::I32Const(i as i32));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
            },
            (Op::Const(Value::Sample(x)), &[]) => {
                // eVeRyThInG is boxed
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                self.dup(wasm::ValType::I32);
                self.insns.push(wasm::Instruction::F32Const(x));
                self.insns.push(wasm::Instruction::F32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
            },
            // TODO: extract these out into a "BinOp" op
            (Op::Add | Op::Sub | Op::Mul | Op::Div, &[e1, e2]) => {
                // TODO: is this the right order?
                self.translate(ctx.clone(), e1);
                self.translate(ctx, e2);
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let t0 = self.temp(wasm::ValType::F32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(match op {
                    Op::Add => wasm::Instruction::F32Add,
                    Op::Sub => wasm::Instruction::F32Sub,
                    Op::Mul => wasm::Instruction::F32Mul,
                    Op::Div => wasm::Instruction::F32Div,
                    _ => unreachable!()
                });
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                let t1 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalTee(t1));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::F32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t1));
            },
            (Op::Sin | Op::Cos, &[e]) => {
                let op_name = match op {
                    Op::Sin => "sin",
                    Op::Cos => "cos",
                    _ => unreachable!(),
                };
                let prim_func = self.translator.runtime_exports[op_name].1;
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                let t0 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalTee(t0));
                self.translate(ctx, e);
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::Call(prim_func));
                self.insns.push(wasm::Instruction::F32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t0));
            },
            (Op::Pi, &[]) => {
                // TODO: SUPER TEMP HACK
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                self.dup(wasm::ValType::I32);
                self.insns.push(wasm::Instruction::F32Const(std::f32::consts::PI));
                self.insns.push(wasm::Instruction::F32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
            },
            (Op::Proj(i), &[e]) => {
                self.translate(ctx, e);
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 4 * i as u64, align: 2, memory_index: 0 }));
            },
            (Op::UnGen, &[e]) => {
                // nop for now
                self.translate(ctx, e);
            },
            (Op::AllocAndFill, _) => {
                self.insns.push(wasm::Instruction::I32Const(args.len() as i32 * 4));
                self.alloc();
                // TODO: this is inefficient
                let l = self.next_local(wasm::ValType::I32);
                self.insns.push(wasm::Instruction::LocalTee(l));
                for (i, &arg) in args.iter().enumerate() {
                    self.translate(ctx.clone(), arg);
                    self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 4 * i as u64, align: 2, memory_index: 0 }));
                    self.insns.push(wasm::Instruction::LocalGet(l));
                }
            },
            (Op::BuildClosure(Global(g)), _) => {
                let g_arity = match self.translator.globals[g as usize] {
                    GlobalDef::Func { arity, .. } => arity,
                    GlobalDef::ClosedExpr { .. } => panic!("shouldn't be seeing closedexprs in buildclosure anymore"),
                };
                self.insns.push(wasm::Instruction::I32Const((args.len() as i32 + 2) * 4));
                self.alloc();
                // TODO: this is inefficient
                let l = self.next_local(wasm::ValType::I32);
                self.insns.push(wasm::Instruction::LocalTee(l));
                self.insns.push(wasm::Instruction::I32Const((self.translator.func_table_offset + g) as i32));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(l));
                self.insns.push(wasm::Instruction::I32Const(g_arity as i32));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(l));
                for (i, &arg) in args.iter().enumerate() {
                    self.translate(ctx.clone(), arg);
                    self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 4 * (i + 2) as u64, align: 2, memory_index: 0 }));
                    self.insns.push(wasm::Instruction::LocalGet(l));
                }
            },
            (Op::LoadGlobal(Global(g)), _) => {
                self.insns.push(wasm::Instruction::GlobalGet(self.translator.globals_offset + g));
            },
            _ =>
                panic!("did not expect {} arguments for op {:?}", args.len(), op)
        }
    }

    fn finish(mut self) -> (u32, wasm::Function) {
        self.locals.drain(..self.arity as usize + 1);
        let mut func = wasm::Function::new_with_locals_types(self.locals);
        for ins in self.insns {
            func.instruction(&ins);
        }
        func.instruction(&wasm::Instruction::End);
        (self.translator.function_types.for_args(self.arity + 1), func)
    }
}
