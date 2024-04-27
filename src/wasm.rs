use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

use string_interner::DefaultStringInterner;
use wasm_encoder as wasm;

use crate::ir1::{DebruijnIndex, Op, Value, Global};
use crate::ir2;
use crate::ir2::{GlobalDef, Expr};

pub fn translate<'a>(global_defs: &[GlobalDef<'a>], main: usize) -> Vec<u8> {
    let mut codes = wasm::CodeSection::new();
    let mut functions = wasm::FunctionSection::new();
    let mut exports = wasm::ExportSection::new();
    exports.export("memory", wasm::ExportKind::Memory, 0);
    let mut function_types = FunctionTypes { types: Vec::new(), back: HashMap::new() };
    let mut globals = wasm::GlobalSection::new();
    let heap_global = global_defs.len() as u32;
    let bad_global = heap_global + 1;
    let mut init_func = wasm::Function::new_with_locals_types([wasm::ValType::I32]);
    for (i, def) in global_defs.into_iter().enumerate() {
        println!("doing {i}");
        globals.global(wasm::GlobalType {
            val_type: wasm::ValType::I32,
            mutable: true,
            shared: false,
        }, &wasm::ConstExpr::i32_const(0));
        match def {
            GlobalDef::Func { rec, arity, env_size, body } => {
                let mut trans = FuncTranslator::new(global_defs, *rec, *env_size, *arity, &mut function_types, heap_global, bad_global);
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
                init_func.instruction(&wasm::Instruction::I32Const(i as i32));
                init_func.instruction(&wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::I32Const(*arity as i32));
                init_func.instruction(&wasm::Instruction::I32Store(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::I32Const(8));
                init_func.instruction(&wasm::Instruction::I32Add);
                init_func.instruction(&wasm::Instruction::GlobalSet(heap_global));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::GlobalSet(i as u32));
            },
            GlobalDef::ClosedExpr { body } => {
                let mut trans = FuncTranslator::new(global_defs, false, 0, 0, &mut function_types, heap_global, bad_global);
                let ctx = trans.make_initial_ctx();
                trans.translate(ctx, body);
                let (type_idx, func) = trans.finish();
                functions.function(type_idx);
                codes.function(&func);

                init_func.instruction(&wasm::Instruction::I32Const(0));
                init_func.instruction(&wasm::Instruction::Call(i as u32));
                init_func.instruction(&wasm::Instruction::GlobalSet(i as u32));
            },
        }
    }


    println!("main is {main}");
    exports.export("main", wasm::ExportKind::Global, main as u32);

    let mut adv_func = wasm::Function::new_with_locals_types([wasm::ValType::I32]);
    let adv_func_type = function_types.for_args(1);
    adv_func.instruction(&wasm::Instruction::LocalGet(0));
    adv_func.instruction(&wasm::Instruction::I32Load(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
    adv_func.instruction(&wasm::Instruction::LocalTee(1));
    adv_func.instruction(&wasm::Instruction::LocalGet(1));
    adv_func.instruction(&wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
    adv_func.instruction(&wasm::Instruction::CallIndirect { ty: adv_func_type, table: 0 });
    adv_func.instruction(&wasm::Instruction::End);
    exports.export("adv_stream", wasm::ExportKind::Func, functions.len());
    functions.function(adv_func_type);
    codes.function(&adv_func);

    let mut hd_func = wasm::Function::new_with_locals_types([]);
    let hd_func_type = function_types.types.len() as u32;
    hd_func.instruction(&wasm::Instruction::LocalGet(0));
    hd_func.instruction(&wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
    hd_func.instruction(&wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
    hd_func.instruction(&wasm::Instruction::End);
    exports.export("hd_stream", wasm::ExportKind::Func, functions.len());
    functions.function(hd_func_type);
    codes.function(&hd_func);

    init_func.instruction(&wasm::Instruction::End);
    let init_func_type = hd_func_type + 1;
    exports.export("init", wasm::ExportKind::Func, functions.len());
    functions.function(init_func_type);
    codes.function(&init_func);

    let mut types = wasm::TypeSection::new();
    for (i, (arg, ret)) in function_types.types.into_iter().enumerate() {
        println!("type #{i}: {arg:?} -> {ret:?}");
        types.function(arg, ret);
    }
    types.function(vec![wasm::ValType::I32], vec![wasm::ValType::F32]);
    types.function(vec![], vec![]);

    let mut memories = wasm::MemorySection::new();
    memories.memory(wasm::MemoryType {
        minimum: 1 << 16,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    globals.global(
        wasm::GlobalType {
            val_type: wasm::ValType::I32,
            mutable: true,
            shared: false,
        },
        &wasm::ConstExpr::i32_const(0)
    );
    globals.global(
        wasm::GlobalType {
            val_type: wasm::ValType::I32,
            mutable: true,
            shared: false,
        },
        &wasm::ConstExpr::i32_const(0)
    );
    exports.export("bad", wasm::ExportKind::Global, bad_global);

    let mut tables = wasm::TableSection::new();
    tables.table(wasm::TableType {
        element_type: wasm::RefType::FUNCREF,
        minimum: global_defs.len() as u32,
        maximum: Some(global_defs.len() as u32),
    });

    let mut elems = wasm::ElementSection::new();
    let function_idxs: Vec<u32> = (0..global_defs.len() as u32).collect();
    let function_elems = wasm::Elements::Functions(&function_idxs);
    elems.active(Some(0), &wasm::ConstExpr::i32_const(0), function_elems);

    let mut module = wasm::Module::new();
    module.section(&types);
    module.section(&functions);
    module.section(&tables);
    module.section(&memories);
    module.section(&globals);
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
    types: Vec<(Vec<wasm::ValType>, Vec<wasm::ValType>)>,
    // #args -> type idx
    back: HashMap<u32, u32>,
}

impl FunctionTypes {
    fn for_args(&mut self, n: u32) -> u32 {
        *self.back.entry(n).or_insert_with(|| {
            let i = self.types.len();
            self.types.push((iter::repeat(wasm::ValType::I32).take(n as usize).collect(), vec![wasm::ValType::I32]));
            i as u32
        })
    }
}

struct FuncTranslator<'a> {
    globals: &'a [GlobalDef<'a>],
    env_size: u32,
    arity: u32,
    locals: Vec<wasm::ValType>,
    insns: Vec<wasm::Instruction<'a>>,
    rec: bool,
    temps: HashMap<(wasm::ValType, u32), u32>,
    function_types: &'a mut FunctionTypes,
    heap_global: u32,
    bad_global: u32,
}

// closure layout: | code addr | arity | used[0] | used[1] | ... | used[n] |

impl<'a> FuncTranslator<'a> {
    fn new(globals: &'a [GlobalDef<'a>], rec: bool, env_size: u32, arity: u32, function_types: &'a mut FunctionTypes, heap_global: u32, bad_global: u32) -> FuncTranslator<'a> {
        FuncTranslator {
            globals,
            env_size,
            rec,
            arity,
            locals: iter::repeat(wasm::ValType::I32).take((env_size+arity+1) as usize).collect(),
            insns: Vec::new(),
            temps: HashMap::new(),
            function_types,
            heap_global,
            bad_global,
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

                /*
                // HACK: while the arity is zero, keep calling the closure
                let loop_ty_idx = self.function_types.for_args(1);
                self.insns.push(wasm::Instruction::Loop(wasm::BlockType::Empty));
                // 
                self.insns.push(wasm::Instruction::LocalGet(t1));
                // i32
                self.insns.push(wasm::Instruction::If(wasm::BlockType::Empty));
                //
                self.insns.push(wasm::Instruction::Else);
                //
                self.insns.push(wasm::Instruction::LocalGet(t0));
                // i32
                self.insns.push(wasm::Instruction::LocalGet(t0));
                // i32 i32
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                // i32 i32
                self.insns.push(wasm::Instruction::CallIndirect { ty: loop_ty_idx, table: 0 });
                // i32
                self.insns.push(wasm::Instruction::LocalTee(t0));
                // i32
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                // i32
                self.insns.push(wasm::Instruction::LocalSet(t1));
                //
                self.insns.push(wasm::Instruction::Br(1));
                self.insns.push(wasm::Instruction::End);
                self.insns.push(wasm::Instruction::End);
                */
                

                self.insns.push(wasm::Instruction::LocalGet(t1));
                self.insns.push(wasm::Instruction::I32Const(args.len() as i32));
                self.insns.push(wasm::Instruction::I32Eq);
                let if_idx = self.function_types.for_args(args.len() as u32);
                self.insns.push(wasm::Instruction::If(wasm::BlockType::FunctionType(if_idx)));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let funty_idx = self.function_types.for_args(args.len() as u32 + 1);
                self.insns.push(wasm::Instruction::CallIndirect { ty: funty_idx, table: 0 });
                self.insns.push(wasm::Instruction::Else);
                // TODO: create closure!
                self.insns.push(wasm::Instruction::I32Const(1));
                self.insns.push(wasm::Instruction::GlobalSet(self.bad_global));
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
        self.insns.push(wasm::Instruction::GlobalGet(self.heap_global));
        let t = self.temp(wasm::ValType::I32, 0);
        self.insns.push(wasm::Instruction::LocalTee(t));
        self.insns.push(wasm::Instruction::I32Add);
        self.insns.push(wasm::Instruction::GlobalSet(self.heap_global));
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
                // TODO: SUPER TEMP HACK
                self.translate(ctx, e);
            },
            //    todo!(),
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
                let g_arity = match self.globals[g as usize] {
                    GlobalDef::Func { arity, .. } => arity,
                    GlobalDef::ClosedExpr { .. } => panic!("shouldn't be seeing closedexprs in buildclosure anymore"),
                };
                self.insns.push(wasm::Instruction::I32Const((args.len() as i32 + 2) * 4));
                self.alloc();
                // TODO: this is inefficient
                let l = self.next_local(wasm::ValType::I32);
                self.insns.push(wasm::Instruction::LocalTee(l));
                self.insns.push(wasm::Instruction::I32Const(g as i32));
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
                self.insns.push(wasm::Instruction::GlobalGet(g));
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
        (self.function_types.for_args(self.arity + 1), func)
    }
}
