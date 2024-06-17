use std::fmt::{self, Write};
use std::error::Error;
use std::collections::HashMap;

use crate::expr::{Expr, Symbol, TopLevelDefBody};
use crate::ir1_egglog::{FromEgglogConverter, ToEgglogConverter};
use egglog::{EGraph, Term, TermDag};
use string_interner::{StringInterner, DefaultStringInterner};

use typed_arena::Arena;

use crate::builtin::{make_builtin_clocks, make_builtins, BuiltinsMap};
use crate::parse::{self, Parser};
use crate::typing::{self, Globals, Typechecker};
use crate::{ir1, ir2, wasm, util};

use crate::typing::Type;

pub struct TopLevel<'a> {
    pub interner: DefaultStringInterner,
    pub arena: &'a Arena<Expr<'a, tree_sitter::Range>>,
    pub builtins: BuiltinsMap,
    pub globals: Globals,
    pub global_clocks: Vec<Symbol>,
}

impl<'a> TopLevel<'a> {
    pub fn new(arena: &'a Arena<Expr<'a, tree_sitter::Range>>) -> TopLevel<'a> {
        let mut interner = StringInterner::new();
        let builtins = make_builtins(&mut interner);
        let globals = builtins.iter().map(|(&name, builtin)| (name, builtin.type_.clone())).collect();
        let builtin_clocks = make_builtin_clocks(&mut interner);

        TopLevel { arena, interner, builtins, globals, global_clocks: builtin_clocks }
    }

    pub fn make_parser<'b>(&'b mut self) -> Parser<'b, 'a> {
        // TODO: this does a bunch of processing we don't need to redo
        Parser::new(&mut self.interner, &mut self.arena)
    }

    pub fn make_typechecker<'b>(&'b mut self) -> Typechecker<'b, 'a, tree_sitter::Range> {
        Typechecker {
            arena: self.arena,
            globals: &mut self.globals,
            global_clocks: &self.global_clocks,
            interner: &mut self.interner,
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // necessary bc we are just printing the debug repr right now
pub enum TopLevelError<'a> {
    IoError(std::io::Error),
    ParseError(String, parse::FullParseError),
    TypeError(String, typing::FileTypeErrors<'a, tree_sitter::Range>),
    InterpError(String),
    CannotSample(Type),
}

impl<'a> fmt::Display for TopLevelError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<'a> Error for TopLevelError<'a> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            TopLevelError::IoError(ref err) => Some(err),
            // TODO: make these errors implement Error
            TopLevelError::ParseError(_, _) => None,
            TopLevelError::TypeError(_, _) => None,
            TopLevelError::InterpError(_) => None,
            TopLevelError::CannotSample(_) => None,
        }
    }
}

impl<'a> From<std::io::Error> for TopLevelError<'a> {
    fn from(err: std::io::Error) -> TopLevelError<'a> {
        TopLevelError::IoError(err)
    }
}

pub type TopLevelResult<'a, T> = Result<T, TopLevelError<'a>>;

// TODO: move this out
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Name {
    Term(Symbol),
    Clock(Symbol),
}

impl Name {
    fn symbol(&self) -> Symbol {
        match *self {
            Name::Term(x) => x,
            Name::Clock(x) => x,
        }
    }
}

fn print_with_intermediate_helper(
    termdag: &TermDag,
    term: Term,
    cache: &mut HashMap<Term, String>,
    res: &mut String,
) -> String {
    if let Some(var) = cache.get(&term) {
        return var.clone();
    }

    match &term {
        Term::Lit(_) => termdag.to_string(&term),
        Term::Var(_) => termdag.to_string(&term),
        Term::App(head, children) => {
            let child_vars = children
                .iter()
                .map(|child| {
                    print_with_intermediate_helper(termdag, termdag.get(*child), cache, res)
                })
                .collect::<Vec<String>>()
                .join(" ");
            let fresh_var = format!("__tmp{}", cache.len());
            writeln!(res, "(let {fresh_var} ({head} {child_vars}))").unwrap();
            cache.insert(term, fresh_var.clone());

            fresh_var
        }
    }
}

pub fn egglog<'a>(toplevel: &mut TopLevel<'a>, code: String) -> TopLevelResult<'a, ()> {
    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let elabbed_file = toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let defs = elabbed_file.defs;

    let mut builtin_globals = HashMap::new();
    let mut global_defs = Vec::new();
    for (&name, builtin) in toplevel.builtins.iter() {
        builtin_globals.insert(name, ir1::Global(global_defs.len() as u32));
        global_defs.push(ir2::GlobalDef::Func {
            rec: false,
            arity: builtin.n_args as u32,
            env_size: 0,
            body: builtin.ir2_expr,
        });
    }

    let mut global_clocks = HashMap::new();
    for &name in toplevel.global_clocks.iter() {
        global_clocks.insert(name, ir1::Global(global_defs.len() as u32));
        global_defs.push(ir2::GlobalDef::ClosedExpr {
            body: &ir2::Expr::Var(ir1::DebruijnIndex(0)),
        });
    }
    for def in defs.iter() {
        match def.body {
            TopLevelDefBody::Def { .. } => {
                builtin_globals.insert(def.name, ir1::Global(global_defs.len() as u32));
            }
            TopLevelDefBody::Clock { .. } => {
                global_clocks.insert(def.name, ir1::Global(global_defs.len() as u32));
            }
        }
        // push a dummy def that we'll replace later, to reserve the space
        global_defs.push(ir2::GlobalDef::ClosedExpr {
            body: &ir2::Expr::Var(ir1::DebruijnIndex(0)),
        });
    }

    let expr_under_arena = Arena::new();
    let expr_ptr_arena = Arena::new();
    let expr_arena = util::ArenaPlus { arena: &expr_under_arena, ptr_arena: &expr_ptr_arena };
    let translator = ir1::Translator { globals: builtin_globals, global_clocks, arena: &expr_arena };

    /*
    let mut egraph: EGraph = Default::default();
    egraph.parse_and_run_program(include_str!("ir1.egg")).unwrap();
    */
    let mut egglog_converter = ToEgglogConverter::new();
    let mut term_cache = HashMap::<Term, String>::new();

    let mut program = String::new();
    
    let _defs_ir1: HashMap<Name, &ir1::Expr<'_>> = toplevel.global_clocks.iter().enumerate().map(|(i, &name)| {
        let clock_expr = &*expr_under_arena.alloc(
            ir1::Expr::Op(ir1::Op::GetClock(i as u32), &[])
        );
        (Name::Clock(name), clock_expr)
    }).chain(defs.iter().map(|def| {
        match def.body {
            TopLevelDefBody::Def { expr, .. } => {
                // println!("compiling {}", toplevel.interner.resolve(def.name).unwrap());
                let expr_ir1 = &*expr_under_arena.alloc(translator.translate(ir1::Ctx::Empty.into(), expr));
                let term = egglog_converter.expr_to_term(expr_ir1);
                let res = print_with_intermediate_helper(&egglog_converter.termdag, term, &mut term_cache, &mut program);
                writeln!(&mut program, "(set (program \"{}\") {})", toplevel.interner.resolve(def.name).unwrap(), res).unwrap();
                (Name::Term(def.name), expr_ir1)
            },
            TopLevelDefBody::Clock { freq } => {
                let clock_expr = &*expr_under_arena.alloc(
                    ir1::Expr::Op(ir1::Op::MakeClock(freq), &[])
                );
                let term = egglog_converter.expr_to_term(clock_expr);
                let res = print_with_intermediate_helper(&egglog_converter.termdag, term, &mut term_cache, &mut program);
                writeln!(program, "(set (clock \"{}\") {})", toplevel.interner.resolve(def.name).unwrap(), res).unwrap();
                (Name::Clock(def.name), clock_expr)
            },
        }
    })).collect();

    let prologue = include_str!("ir1.egg");
    let full_program = format!("
{prologue}

{program}

(run 1)
    ");

    let mut egraph: EGraph = Default::default();
    egraph.parse_and_run_program(&full_program).unwrap();

    let mut from_converter = FromEgglogConverter {
        termdag: Default::default(),
        arena: &expr_arena,
        expr_cache: Default::default(),
        slice_cache: Default::default(),
    };

    for def in defs.iter() {
        let name = toplevel.interner.resolve(def.name).unwrap();
        let kind = match def.body {
            TopLevelDefBody::Def { .. } => "program",
            TopLevelDefBody::Clock { .. } => "clock",
        };
        let (sort, val) = egraph.eval_expr(&egglog::ast::Expr::Call((), kind.into(), vec![egglog::ast::Expr::Lit((), egglog::ast::Literal::String(name.into()))])).unwrap();
        let (_, extracted) = egraph.extract(val, &mut from_converter.termdag, &sort);
        println!("{}", from_converter.termdag.to_string(&extracted));
        let expr = from_converter.term_to_expr(extracted);
        println!("{}: {:?}", name, expr);
    }

    Ok(())
}

pub fn compile<'a>(toplevel: &mut TopLevel<'a>, code: String) -> TopLevelResult<'a, Vec<u8>> {
    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let elabbed_file = toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let defs = elabbed_file.defs;

    let mut builtin_globals = HashMap::new();
    let mut global_defs = Vec::new();
    for (&name, builtin) in toplevel.builtins.iter() {
        builtin_globals.insert(name, ir1::Global(global_defs.len() as u32));
        global_defs.push(ir2::GlobalDef::Func {
            rec: false,
            arity: builtin.n_args as u32,
            env_size: 0,
            body: builtin.ir2_expr,
        });
    }

    let mut global_clocks = HashMap::new();
    for &name in toplevel.global_clocks.iter() {
        global_clocks.insert(name, ir1::Global(global_defs.len() as u32));
        global_defs.push(ir2::GlobalDef::ClosedExpr {
            body: &ir2::Expr::Var(ir1::DebruijnIndex(0)),
        });
    }
    for def in defs.iter() {
        match def.body {
            TopLevelDefBody::Def { .. } => {
                builtin_globals.insert(def.name, ir1::Global(global_defs.len() as u32));
            }
            TopLevelDefBody::Clock { .. } => {
                global_clocks.insert(def.name, ir1::Global(global_defs.len() as u32));
            }
        }
        // push a dummy def that we'll replace later, to reserve the space
        global_defs.push(ir2::GlobalDef::ClosedExpr {
            body: &ir2::Expr::Var(ir1::DebruijnIndex(0)),
        });
    }

    let expr_under_arena = Arena::new();
    let expr_ptr_arena = Arena::new();
    let expr_arena = util::ArenaPlus { arena: &expr_under_arena, ptr_arena: &expr_ptr_arena };
    let translator = ir1::Translator { globals: builtin_globals, global_clocks, arena: &expr_arena };

    let mut egglog_converter = ToEgglogConverter::new();
    let mut term_cache = HashMap::<Term, String>::new();

    let mut program = String::new();

    let mut unopt_expr_ir1 = HashMap::new();

    for def in defs.iter() {
        match def.body {
            TopLevelDefBody::Def { expr, .. } => {
                // println!("compiling {}", toplevel.interner.resolve(def.name).unwrap());
                let expr_ir1 = &*expr_under_arena.alloc(translator.translate(ir1::Ctx::Empty.into(), expr));
                unopt_expr_ir1.insert(def.name, expr_ir1);
                let term = egglog_converter.expr_to_term(expr_ir1);
                let res = print_with_intermediate_helper(&egglog_converter.termdag, term, &mut term_cache, &mut program);
                writeln!(&mut program, "(set (program \"{}\") {})", toplevel.interner.resolve(def.name).unwrap(), res).unwrap();
            },
            TopLevelDefBody::Clock { freq } => {
                let clock_expr = &*expr_under_arena.alloc(
                    ir1::Expr::Op(ir1::Op::MakeClock(freq), &[])
                );
                let term = egglog_converter.expr_to_term(clock_expr);
                let res = print_with_intermediate_helper(&egglog_converter.termdag, term, &mut term_cache, &mut program);
                writeln!(program, "(set (clock \"{}\") {})", toplevel.interner.resolve(def.name).unwrap(), res).unwrap();
            },
        }
    }

    let prologue = include_str!("ir1.egg");
    let full_program = format!("
{prologue}

{program}

(run 1)
    ");

    let mut egraph: EGraph = Default::default();
    egraph.parse_and_run_program(&full_program).unwrap();

    let mut from_converter = FromEgglogConverter {
        termdag: Default::default(),
        arena: &expr_arena,
        expr_cache: Default::default(),
        slice_cache: Default::default(),
    };
    
    let mut main = None;
    let opt_defs_ir1: HashMap<Name, &ir1::Expr<'_>> = toplevel.global_clocks.iter().enumerate().map(|(i, &name)| {
        let clock_expr = &*expr_under_arena.alloc(
            ir1::Expr::Op(ir1::Op::GetClock(i as u32), &[])
        );
        (Name::Clock(name), clock_expr)
    }).chain(defs.iter().map(|def| {
        match def.body {
            TopLevelDefBody::Def { expr: _orig_expr, .. } => {
                let name = toplevel.interner.resolve(def.name).unwrap();
                let kind = "program";
                let (sort, val) = egraph.eval_expr(&egglog::ast::Expr::Call((), kind.into(), vec![egglog::ast::Expr::Lit((), egglog::ast::Literal::String(name.into()))])).unwrap();
                let (_, extracted) = egraph.extract(val, &mut from_converter.termdag, &sort);
                let opt_expr = from_converter.term_to_expr(extracted);
                let unopt_expr = unopt_expr_ir1[&def.name];
                println!("unoptimized {name}: {unopt_expr:?}");
                println!("optimized {name}: {opt_expr:?}");
                let (annotated, _) = translator.annotate_used_vars(opt_expr);
                let shifted = translator.shift(annotated, 0, 0, &imbl::HashMap::new());
                (Name::Term(def.name), shifted)
            },
            TopLevelDefBody::Clock { freq } => {
                let clock_expr = &*expr_under_arena.alloc(
                    ir1::Expr::Op(ir1::Op::MakeClock(freq), &[])
                );
                (Name::Clock(def.name), clock_expr)
            },
        }
    })).collect();

    let expr2_under_arena = Arena::new();
    let expr2_ptr_arena = Arena::new();
    let expr2_arena = util::ArenaPlus { arena: &expr2_under_arena, ptr_arena: &expr2_ptr_arena };
    let mut translator2 = ir2::Translator { arena: &expr2_arena, globals: global_defs };

    for (name, expr) in opt_defs_ir1 {
        let expr_ir2 = translator2.translate(expr);
        let def_idx = match name {
            Name::Term(sym) => translator.globals[&sym].0 as usize,
            Name::Clock(sym) => translator.global_clocks[&sym].0 as usize,
        };
        if name == Name::Term(toplevel.interner.get_or_intern_static("main")) {
            main = Some(def_idx);
        }
        translator2.globals[def_idx] = ir2::GlobalDef::ClosedExpr { body: expr_ir2 };
        println!("{}: {:?}", toplevel.interner.resolve(name.symbol()).unwrap(), expr_ir2);
    }

    let mut global_defs = translator2.globals;
    for (i, func) in global_defs.iter().enumerate() {
        println!("global {i}: {func:?}");
    }

    // generate partial application functions
    let partial_app_def_offset = global_defs.len() as u32;
    let max_arity = global_defs.iter().map(|def| def.arity().unwrap_or(0)).max().unwrap();
    // let partial_app_defs = Vec::with_capacity(max_arity * (max_arity + 1) / 2 - max_arity);
    for arity in 2..=max_arity {
        for n_args in 1..arity {
            let n_remaining_args = arity - n_args;
            let args_to_call = (n_remaining_args..arity).map(|i| {
                expr2_arena.alloc(ir2::Expr::Var(ir1::DebruijnIndex(i+1)))
            }).chain((0..n_remaining_args).map(|i| {
                // expr2_arena.alloc(ir2::Expr::Op(ir1::Op::UnboxedConst(ir1::Value::Index(42+i as usize)), &[]))
                expr2_arena.alloc(ir2::Expr::Var(ir1::DebruijnIndex(n_remaining_args - 1 - i)))
            }));
            global_defs.push(ir2::GlobalDef::Func {
                rec: false,
                arity: n_remaining_args,
                env_size: n_args + 1,
                body: expr2_arena.alloc(ir2::Expr::CallIndirect(
                    expr2_arena.alloc(ir2::Expr::Var(ir1::DebruijnIndex(n_remaining_args))),
                    expr2_arena.alloc_slice_r(args_to_call)
                )),
            });
        }
    }

    let wasm_bytes = wasm::translate(&global_defs, partial_app_def_offset, main.unwrap());

    Ok(wasm_bytes)
}

#[cfg(feature="run")]
pub fn run(wasm_bytes: &[u8], num_samples: usize) -> Vec<f32> {
    use byteorder::{ReadBytesExt, LittleEndian};

    let engine = wasmtime::Engine::new(
        &wasmtime::Config::new()
            .profiler(wasmtime::ProfilingStrategy::PerfMap)
    ).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm_bytes).unwrap();
    let linker = wasmtime::Linker::new(&engine);
    let mut store: wasmtime::Store<()> = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();

    let alloc = instance.get_typed_func::<u32, u32>(&mut store, "alloc").unwrap();
    let sample_scheduler = instance.get_typed_func::<(u32, u32, u32), u32>(&mut store, "sample_scheduler").unwrap();

    let samples_ptr = alloc.call(&mut store, 4 * 128).unwrap();
    let mut main = instance.get_global(&mut store, "main").unwrap().get(&mut store).unwrap_i32() as u32;

    const CHUNK_SIZE: usize = 128;
    let num_chunks = (num_samples + CHUNK_SIZE - 1) / CHUNK_SIZE;
    let mut all_samples = Vec::with_capacity(num_chunks * CHUNK_SIZE);
    for _ in 0..num_chunks {
        main = sample_scheduler.call(&mut store, (main, CHUNK_SIZE as u32, samples_ptr)).unwrap();
        let mut samples_buf = [0u8; CHUNK_SIZE*4];
        memory.read(&mut store, samples_ptr as usize, &mut samples_buf).unwrap();
        // could almost certainly make this more efficient, but it's
        // not the bottleneck here. let's not worry about it for now.
        let mut remaining_samples = &samples_buf[..];
        while remaining_samples.len() > 0 {
            all_samples.push(remaining_samples.read_f32::<LittleEndian>().unwrap());
        }
    }

    all_samples
}
