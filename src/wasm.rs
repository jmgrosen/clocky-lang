use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

use wasm::FuncType;
use wasm_encoder as wasm;
use indexmap::IndexSet;

use crate::ir1::{DebruijnIndex, Op, Value, Global};
use crate::ir2::{GlobalDef, Expr};
use crate::runtime::Runtime;

const RUNTIME_BYTES: &'static [u8] = include_bytes!(env!("CARGO_CDYLIB_FILE_CLOCKY_RUNTIME"));

pub fn translate<'a>(global_defs: &[GlobalDef<'a>], partial_app_def_offset: u32, main: usize) -> Vec<u8> {
    // TODO: can we parse more of this at compile time?
    // probably... would have to be a build script though, I imagine
    let runtime = Runtime::from_bytes(RUNTIME_BYTES);

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
        partial_app_table_offset: partial_app_def_offset + func_offset,
        function_types: &mut function_types,
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
                let mut trans = FuncTranslator::new(&mut translator, *rec, *env_size, *arity);
                let ctx = trans.make_initial_ctx();
                trans.translate(ctx, body);
                let (type_idx, func) = trans.finish();
                functions.function(type_idx);
                codes.function(&func);

                init_func.instruction(&wasm::Instruction::I32Const(8));
                init_func.instruction(&wasm::Instruction::Call(translator.runtime_exports["alloc"].1));
                init_func.instruction(&wasm::Instruction::LocalTee(0));
                init_func.instruction(&wasm::Instruction::I32Const(func_offset as i32 + i as i32));
                init_func.instruction(&wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::I32Const(*arity as i32));
                init_func.instruction(&wasm::Instruction::I32Store(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                init_func.instruction(&wasm::Instruction::LocalGet(0));
                init_func.instruction(&wasm::Instruction::GlobalSet(globals_offset + i as u32));
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
        minimum: runtime.initial_memory_size as u64,
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
    runtime.emit_elements(&mut elems);
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
    partial_app_table_offset: u32,
    function_types: &'a mut FunctionTypes,
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
                let call_target_closure = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalTee(call_target_closure));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                let arity = self.temp(wasm::ValType::I32, 1);
                self.insns.push(wasm::Instruction::LocalTee(arity));

                // TODO: should this logic be outsourced to a single
                // function per number of args? probably
                self.insns.push(wasm::Instruction::I32Const(args.len() as i32));
                self.insns.push(wasm::Instruction::I32Eq);
                let if_idx = self.translator.function_types.for_args(args.len() as u32);
                self.insns.push(wasm::Instruction::If(wasm::BlockType::FunctionType(if_idx)));

                self.insns.push(wasm::Instruction::LocalGet(call_target_closure));
                self.insns.push(wasm::Instruction::LocalGet(call_target_closure));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let funty_idx = self.translator.function_types.for_args(args.len() as u32 + 1);
                self.insns.push(wasm::Instruction::CallIndirect { ty: funty_idx, table: 0 });

                self.insns.push(wasm::Instruction::Else);

                // we'll have to create a partial application closure.
                let closure_size = (args.len() as i32 + 3) * 4;
                self.insns.push(wasm::Instruction::I32Const(closure_size));
                self.alloc();
                let partial_app_closure = self.temp(wasm::ValType::I32, 2);
                self.insns.push(wasm::Instruction::LocalTee(partial_app_closure));

                // TODO: move this calculation logic somewhere else to couple this less badly.
                //
                // if n is the arity of the closure we're trying to
                // call and m is the number of arguments we want to
                // apply (where m < n), the index of the function
                // should be (((n-1) * (n-2))/2 + (m - 1))
                self.insns.push(wasm::Instruction::LocalGet(arity));
                self.insns.push(wasm::Instruction::I32Const(1));
                self.insns.push(wasm::Instruction::I32Sub);
                self.insns.push(wasm::Instruction::LocalGet(arity));
                self.insns.push(wasm::Instruction::I32Const(2));
                self.insns.push(wasm::Instruction::I32Sub);
                self.insns.push(wasm::Instruction::I32Mul);
                self.insns.push(wasm::Instruction::I32Const(1));
                self.insns.push(wasm::Instruction::I32ShrU);
                let offset = self.translator.partial_app_table_offset as i32 + args.len() as i32 - 1;
                self.insns.push(wasm::Instruction::I32Const(offset));
                self.insns.push(wasm::Instruction::I32Add);
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));

                self.insns.push(wasm::Instruction::LocalGet(partial_app_closure));
                self.insns.push(wasm::Instruction::LocalGet(arity));
                self.insns.push(wasm::Instruction::I32Const(args.len() as i32));
                self.insns.push(wasm::Instruction::I32Sub);
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));

                self.insns.push(wasm::Instruction::LocalGet(partial_app_closure));
                self.insns.push(wasm::Instruction::LocalGet(call_target_closure));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 8, align: 2, memory_index: 0 }));

                let arg_temp = self.temp(wasm::ValType::I32, 3);
                for i in 0..args.len() {
                    self.insns.push(wasm::Instruction::LocalSet(arg_temp));
                    self.insns.push(wasm::Instruction::LocalGet(partial_app_closure));
                    self.insns.push(wasm::Instruction::LocalGet(arg_temp));
                    self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 12 + (i as u64) * 4, align: 2, memory_index: 0 }));
                }

                self.insns.push(wasm::Instruction::LocalGet(partial_app_closure));

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
        self.insns.push(wasm::Instruction::Call(self.translator.runtime_exports["alloc"].1));
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
            // TODO: extract these out into a "BinOp" op?
            (Op::FAdd | Op::FSub | Op::FMul | Op::FDiv, &[e1, e2]) => {
                // TODO: is this the right order?
                self.translate(ctx.clone(), e1);
                self.translate(ctx, e2);
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let t0 = self.temp(wasm::ValType::F32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(match op {
                    Op::FAdd => wasm::Instruction::F32Add,
                    Op::FSub => wasm::Instruction::F32Sub,
                    Op::FMul => wasm::Instruction::F32Mul,
                    Op::FDiv => wasm::Instruction::F32Div,
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
            (Op::FGt | Op::FGe | Op::FLt | Op::FLe | Op::FEq | Op::FNe, &[e1, e2]) => {
                // TODO: is this the right order?
                //
                // TODO: obviously it is silly to allocate a fresh
                // boolean for every comparison. we should statically
                // allocate two booleans.
                self.translate(ctx.clone(), e1);
                self.translate(ctx, e2);
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let t0 = self.temp(wasm::ValType::F32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(match op {
                    Op::FGt => wasm::Instruction::F32Gt,
                    Op::FGe => wasm::Instruction::F32Ge,
                    Op::FLt => wasm::Instruction::F32Lt,
                    Op::FLe => wasm::Instruction::F32Le,
                    Op::FEq => wasm::Instruction::F32Eq,
                    Op::FNe => wasm::Instruction::F32Ne,
                    _ => unreachable!()
                });
                let t1 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t1));
                self.insns.push(wasm::Instruction::I32Const(8));
                self.alloc();
                let t2 = self.temp(wasm::ValType::I32, 1);
                self.insns.push(wasm::Instruction::LocalTee(t2));
                self.insns.push(wasm::Instruction::LocalGet(t1));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t2));
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
            (Op::IAdd | Op::ISub | Op::IMul | Op::IDiv | Op::Shl | Op::Shr | Op::And | Op::Xor | Op::Or, &[e1, e2]) => {
                // TODO: is this the right order?
                //
                self.translate(ctx.clone(), e1);
                self.translate(ctx, e2);
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let t0 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(match op {
                    Op::IAdd => wasm::Instruction::I32Add,
                    Op::ISub => wasm::Instruction::I32Sub,
                    Op::IMul => wasm::Instruction::I32Mul,
                    Op::IDiv => wasm::Instruction::I32DivU,
                    Op::Shl => wasm::Instruction::I32Shl,
                    Op::Shr => wasm::Instruction::I32ShrU,
                    Op::And => wasm::Instruction::I32And,
                    Op::Xor => wasm::Instruction::I32Xor,
                    Op::Or => wasm::Instruction::I32Or,
                    _ => unreachable!()
                });
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                let t1 = self.temp(wasm::ValType::I32, 1);
                self.insns.push(wasm::Instruction::LocalTee(t1));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t1));
            },
            (Op::IGt | Op::IGe | Op::ILt | Op::ILe | Op::IEq | Op::INe, &[e1, e2]) => {
                // TODO: is this the right order?
                //
                // TODO: obviously it is silly to allocate a fresh
                // boolean for every comparison. we should statically
                // allocate two booleans.
                self.translate(ctx.clone(), e1);
                self.translate(ctx, e2);
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let t0 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(match op {
                    Op::IGt => wasm::Instruction::I32GtU,
                    Op::IGe => wasm::Instruction::I32GeU,
                    Op::ILt => wasm::Instruction::I32LtU,
                    Op::ILe => wasm::Instruction::I32LeU,
                    Op::IEq => wasm::Instruction::I32Eq,
                    Op::INe => wasm::Instruction::I32Ne,
                    _ => unreachable!()
                });
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::I32Const(8));
                self.alloc();
                let t1 = self.temp(wasm::ValType::I32, 1);
                self.insns.push(wasm::Instruction::LocalTee(t1));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t1));
            },
            (Op::ReinterpF2I, &[e]) => {
                self.translate(ctx, e);
                self.insns.push(wasm::Instruction::F32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::I32ReinterpretF32);
                let t0 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                let t1 = self.temp(wasm::ValType::I32, 1);
                self.insns.push(wasm::Instruction::LocalTee(t1));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t1));
            },
            (Op::ReinterpI2F | Op::CastI2F, &[e]) => {
                self.translate(ctx, e);
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(match op {
                    Op::ReinterpI2F => wasm::Instruction::F32ReinterpretI32,
                    Op::CastI2F => wasm::Instruction::F32ConvertI32U,
                    _ => unreachable!(),
                });
                let t0 = self.temp(wasm::ValType::F32, 0);
                self.insns.push(wasm::Instruction::LocalSet(t0));
                self.insns.push(wasm::Instruction::I32Const(4));
                self.alloc();
                let t1 = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalTee(t1));
                self.insns.push(wasm::Instruction::LocalGet(t0));
                self.insns.push(wasm::Instruction::F32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(t1));
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
            (Op::ApplyCoeff(_coeff), &[clock]) => {
                // TODO: actually apply the coefficient
                self.translate(ctx, clock);
            },
            (Op::SinceLastTickStream, &[clock]) => {
                self.translate(ctx, clock);
                self.insns.push(wasm::Instruction::Call(self.translator.runtime_exports["since_last_tick_stream"].1));
            },
            (Op::Advance, &[delayed]) => {
                self.translate(ctx, delayed);
                // on the stack now is a pointer to either a closure
                // or a value, offsetted. essentially a thunk. we
                // check which one by seeing if the first word is 0.
                let thunk = self.temp(wasm::ValType::I32, 0);
                self.insns.push(wasm::Instruction::LocalTee(thunk));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                let func = self.temp(wasm::ValType::I32, 1);
                self.insns.push(wasm::Instruction::LocalTee(func));

                self.insns.push(wasm::Instruction::If(wasm::BlockType::Result(wasm::ValType::I32)));

                self.insns.push(wasm::Instruction::LocalGet(thunk));
                self.insns.push(wasm::Instruction::LocalGet(func));
                let funty_idx = self.translator.function_types.for_args(1);
                self.insns.push(wasm::Instruction::CallIndirect { ty: funty_idx, table: 0 });
                let result = self.temp(wasm::ValType::I32, 2);
                self.insns.push(wasm::Instruction::LocalSet(result));
                self.insns.push(wasm::Instruction::LocalGet(thunk));
                self.insns.push(wasm::Instruction::I32Const(0));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 0, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(thunk));
                self.insns.push(wasm::Instruction::LocalGet(result));
                self.insns.push(wasm::Instruction::I32Store(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));
                self.insns.push(wasm::Instruction::LocalGet(result));

                self.insns.push(wasm::Instruction::Else);

                self.insns.push(wasm::Instruction::LocalGet(thunk));
                self.insns.push(wasm::Instruction::I32Load(wasm::MemArg { offset: 4, align: 2, memory_index: 0 }));

                self.insns.push(wasm::Instruction::End);
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
