use core::fmt;
use std::error::Error;
use std::path::Path;
use std::process::ExitCode;
use std::{collections::HashMap, path::PathBuf, fs::File};
use std::io::{Read, Write};

use clocky::expr::{Expr, Symbol};
use num::One;
use string_interner::{StringInterner, DefaultStringInterner};

use typed_arena::Arena;

use clap::Parser as CliParser;

use clocky::builtin::make_builtins;
use clocky::interp::get_samples;
use clocky::parse::{self, Parser};
use clocky::typing::{self, Globals, Typechecker, Ctx};
use clocky::{ir1, ir2, interp, wasm, util};

use clocky::typing::{Type, TopLevelTypeError, Kind, Clock};

#[derive(CliParser, Debug)]
struct Args {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    /// Parse the given program, printing the attempted parse tree if unable
    Parse {
        file: Option<PathBuf>,

        /// Dump the dot graph of the tree sitter parse tree
        #[arg(long)]
        dump_to: Option<PathBuf>,
    },
    /// Type-check the given program.
    Typecheck {
        /// Code file to use
        file: Option<PathBuf>,
    },
    /// Interpret the given program.
    Interpret {
        /// Code file to use
        file: Option<PathBuf>,
    },
    /// Launch a REPL
    Repl,
    /// Write out a WAV file, sampling the stream specified in the code
    Sample {
        /// Path to WAV file to be written
        #[arg(short='o')]
        out: PathBuf,

        /// How long (in milliseconds at 48kHz) to sample for
        #[arg(short='l', default_value_t=1000)]
        length: usize,

        /// Code file to use
        file: Option<PathBuf>,
    },
    Compile {
        /// Code file to use
        file: Option<PathBuf>,

        /// Path to wasm module file to be written
        #[arg(short='o')]
        out: Option<PathBuf>,
    },
}

fn read_file(name: Option<&Path>) -> std::io::Result<String> {
    let mut s = String::new();
    match name {
        Some(path) => { File::open(path)?.read_to_string(&mut s)?; },
        None => { std::io::stdin().read_to_string(&mut s)?; }
    }
    Ok(s)
}

fn write_file(name: Option<&Path>, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(path) = name {
        File::create(path)?.write_all(bytes)
    } else {
        // for now, just don't write it. eventually write to stdout
        Ok(())
    }
}

struct TopLevel<'a> {
    interner: DefaultStringInterner,
    arena: &'a Arena<Expr<'a, tree_sitter::Range>>,
    globals: Globals,
}

impl<'a> TopLevel<'a> {
    fn make_parser<'b>(&'b mut self) -> Parser<'b, 'a> {
        // TODO: this does a bunch of processing we don't need to redo
        Parser::new(&mut self.interner, &mut self.arena)
    }

    fn make_typechecker<'b>(&'b mut self) -> Typechecker<'b, 'a, tree_sitter::Range> {
        Typechecker { arena: self.arena,  globals: &mut self.globals, interner: &mut self.interner }
    }
}

#[derive(Debug)]
#[allow(dead_code)] // necessary bc we are just printing the debug repr right now
enum TopLevelError<'a> {
    IoError(std::io::Error),
    ParseError(String, parse::FullParseError),
    TypeError(String, typing::FileTypeErrors<'a, tree_sitter::Range>),
    InterpError(String),
    CannotSample(Type),
    WavError(hound::Error),
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
            TopLevelError::WavError(ref err) => Some(err),
        }
    }
}

impl<'a> From<std::io::Error> for TopLevelError<'a> {
    fn from(err: std::io::Error) -> TopLevelError<'a> {
        TopLevelError::IoError(err)
    }
}

impl<'a> From<hound::Error> for TopLevelError<'a> {
    fn from(err: hound::Error) -> TopLevelError<'a> {
        TopLevelError::WavError(err)
    }
}

type TopLevelResult<'a, T> = Result<T, TopLevelError<'a>>;

fn cmd_parse<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>, dump_to: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => {
            println!("{}", parsed_file.pretty(&toplevel.interner));
        },
        Err(parse::FullParseError { tree, error }) => {
            eprintln!("{:?}", error);
            if let Some(dump_path) = dump_to {
                let dump_file = File::create(dump_path)?;
                #[cfg(not(target_arch = "wasm32"))]
                tree.print_dot_graph(&dump_file);
            }
        },
    }
    Ok(())
}

fn cmd_typecheck<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;
    Ok(())
}

fn cmd_interpret<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let elabbed_file = toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let arena = Arena::new();
    let defs: HashMap<Symbol, &Expr<'_, ()>> = elabbed_file.defs.iter().map(|def| (def.name, &*arena.alloc(def.body.map_ext(&arena, &(|_| ()))))).collect();
    let main = *defs.get(&parsed_file.defs.last().unwrap().name).unwrap();
    let builtins = make_builtins(&mut toplevel.interner);
    let interp_ctx = interp::InterpretationContext { builtins: &builtins, defs: &defs, env: HashMap::new() };

    match interp::interp(&interp_ctx, &main) {
        Ok(v) => println!("{:?}", v),
        Err(e) => return Err(TopLevelError::InterpError(e.into())),
    }

    Ok(())
}

fn repl_one<'a>(toplevel: &mut TopLevel<'a>, interp_ctx: &interp::InterpretationContext<'a, '_>, code: String) -> TopLevelResult<'a, ()> {
    // this seems like a hack, but it seems like tree-sitter doesn't
    // support parsing specific rules yet...
    let code = format!("def main: unit = {}", code);
    let expr = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file.defs[0].body,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let empty_symbol = toplevel.interner.get_or_intern_static("");
    let (expr_elab, ty) = toplevel
        .make_typechecker()
        .synthesize(&Ctx::Empty, expr)
        .map_err(|e| TopLevelError::TypeError(code, typing::FileTypeErrors { errs: vec![TopLevelTypeError::TypeError(empty_symbol, e)] } ))?;
    println!("synthesized type: {}", ty.pretty(&toplevel.interner));

    let arena = Arena::new();
    let expr_unannotated = expr_elab.map_ext(&arena, &(|_| ()));

    match interp::interp(&interp_ctx, &expr_unannotated) {
        Ok(v) => println!("{:?}", v),
        Err(e) => return Err(TopLevelError::InterpError(e.into())),
    }

    Ok(())
}

fn cmd_repl<'a>(toplevel: &mut TopLevel<'a>) -> TopLevelResult<'a, ()> {
    let builtins = make_builtins(&mut toplevel.interner);
    let interp_ctx = interp::InterpretationContext { builtins: &builtins, defs: &HashMap::new(), env: HashMap::new() };

    for line in std::io::stdin().lines() {
        match repl_one(toplevel, &interp_ctx, line?) {
            Ok(()) => { },
            Err(err @ TopLevelError::IoError(_)) => { return Err(err); },
            Err(TopLevelError::TypeError(code, e)) => {
                let TopLevelTypeError::TypeError(_, ref err) = e.errs[0] else { panic!() };
                println!("{}", err.pretty(&toplevel.interner, &code));
            },
            Err(err) => { println!("{:?}", err); },
        }
    }

    Ok(())
}

fn verify_sample_type<'a>(type_: &Type) -> TopLevelResult<'a, ()> {
    match *type_ {
        Type::Forall(x, Kind::Clock, ref st) =>
            match **st {
                Type::Stream(Clock { coeff, var }, ref inner)
                    if x == var && coeff.is_one() =>
                    match **inner {
                        Type::Sample =>
                            Ok(()),
                        _ => Err(TopLevelError::CannotSample(type_.clone())),
                    },
                _ =>
                    Err(TopLevelError::CannotSample(type_.clone())),
            },
        _ =>
            Err(TopLevelError::CannotSample(type_.clone())),
    }
}

fn cmd_sample<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>, length: usize, out: PathBuf) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let elabbed_file = toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let arena = Arena::new();
    let defs: HashMap<Symbol, &Expr<'_, ()>> = elabbed_file.defs.iter().map(|def| (def.name, &*arena.alloc(def.body.map_ext(&arena, &(|_| ()))))).collect();
    let main_sym = toplevel.interner.get_or_intern_static("main");
    let main_def = parsed_file.defs.iter().filter(|x| x.name == main_sym).next().unwrap();
    verify_sample_type(&main_def.type_)?;
    let main = *defs.get(&parsed_file.defs.last().unwrap().name).unwrap();
    let builtins = make_builtins(&mut toplevel.interner);
    let mut samples = [0.0].repeat(48 * length);

    match get_samples(&builtins, &defs, main, &mut samples[..]) {
        Ok(()) => { },
        Err(e) => return Err(TopLevelError::InterpError(e)),
    }

    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut hound_writer = hound::WavWriter::create(out, wav_spec)?;
    // is this really the only way hound will let me do this??
    for samp in samples {
        hound_writer.write_sample(samp)?;
    }

    Ok(())
}

fn cmd_compile<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>, out: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let elabbed_file = toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let arena = Arena::new();
    let defs: Vec<(Symbol, &Expr<'_, ()>)> = elabbed_file.defs.iter().map(|def| (def.name, &*arena.alloc(def.body.map_ext(&arena, &(|_| ()))))).collect();

    let builtins = make_builtins(&mut toplevel.interner);
    let mut builtin_globals = HashMap::new();
    let mut global_defs = Vec::new();
    for (name, builtin) in builtins.into_iter() {
        builtin_globals.insert(name, ir1::Global(global_defs.len() as u32));
        global_defs.push(ir2::GlobalDef::Func {
            rec: false,
            arity: builtin.n_args as u32,
            env_size: 0,
            body: builtin.ir2_expr,
        });
    }

    for &(name, _) in defs.iter() {
        builtin_globals.insert(name, ir1::Global(global_defs.len() as u32));
        // push a dummy def that we'll replace later, to reserve the space
        global_defs.push(ir2::GlobalDef::ClosedExpr {
            body: &ir2::Expr::Var(ir1::DebruijnIndex(0)),
        });
    }

    let expr_under_arena = Arena::new();
    let expr_ptr_arena = Arena::new();
    let expr_arena = util::ArenaPlus { arena: &expr_under_arena, ptr_arena: &expr_ptr_arena };
    let translator = ir1::Translator { globals: builtin_globals, arena: &expr_arena };

    
    let mut main = None;
    let defs_ir1: HashMap<Symbol, &ir1::Expr<'_>> = defs.iter().map(|&(name, expr)| {
        let expr_ir1 = expr_under_arena.alloc(translator.translate(ir1::Ctx::Empty.into(), expr));
        let (annotated, _) = translator.annotate_used_vars(expr_ir1);
        let shifted = translator.shift(annotated, 0, 0, &imbl::HashMap::new());
        (name, shifted)
    }).collect();

    let expr2_under_arena = Arena::new();
    let expr2_ptr_arena = Arena::new();
    let expr2_arena = util::ArenaPlus { arena: &expr2_under_arena, ptr_arena: &expr2_ptr_arena };
    let mut translator2 = ir2::Translator { arena: &expr2_arena, globals: global_defs };

    for (name, expr) in defs_ir1 {
        let expr_ir2 = translator2.translate(expr);
        let def_idx = translator.globals[&name].0 as usize;
        if name == toplevel.interner.get_or_intern_static("main") {
            println!("found main: {def_idx}");
            main = Some(def_idx);
        }
        translator2.globals[def_idx] = ir2::GlobalDef::ClosedExpr { body: expr_ir2 };
    }

    for (i, func) in translator2.globals.iter().enumerate() {
        println!("global {i}: {func:?}");
    }

    let wasm_bytes = wasm::translate(&translator2.globals, main.unwrap());
    let orig_wasm_bytes = wasm_bytes.clone();

    write_file(out.as_deref(), &wasm_bytes)?;

    //wasmparser::validate(&wasm_bytes).unwrap();

    run(orig_wasm_bytes);

    Ok(())
}

#[cfg(feature="run")]
fn run(mut wasm_bytes: Vec<u8>) {
    let wasm_mod = wasm_bytes.clone();
    use wasmparser::{Chunk, Payload::*};
    let mut validator = wasmparser::Validator::new();
    let mut cur = wasmparser::Parser::new(0);
    let mut stack = Vec::new();

    loop {
        let (payload, consumed) = match cur.parse(&wasm_bytes, false).unwrap() {
            Chunk::NeedMoreData(_) => {
                break;
            }

            Chunk::Parsed { consumed, payload } => (payload, consumed),
        };

        println!("{payload:?}");
        match validator.payload(&payload).unwrap() {
            wasmparser::ValidPayload::Func(func, body) => {
                let mut v = func.into_validator(Default::default());
                v.validate(&body).unwrap();
            },
            _ => {}
        }
        match payload {
            // Sections for WebAssembly modules
            Version { .. } => { /* ... */ }
            TypeSection(_) => { /* ... */ }
            ImportSection(_) => { /* ... */ }
            FunctionSection(_) => { /* ... */ }
            TableSection(_) => { /* ... */ }
            MemorySection(_) => { /* ... */ }
            TagSection(_) => { /* ... */ }
            GlobalSection(_) => { /* ... */ }
            ExportSection(_) => { /* ... */ }
            StartSection { .. } => { /* ... */ }
            ElementSection(_) => { /* ... */ }
            DataCountSection { .. } => { /* ... */ }
            DataSection(_) => { /* ... */ }

            // Here we know how many functions we'll be receiving as
            // `CodeSectionEntry`, so we can prepare for that, and
            // afterwards we can parse and handle each function
            // individually.
            CodeSectionStart { .. } => { /* ... */ }
            CodeSectionEntry(_) => {
                // here we can iterate over `body` to parse the function
                // and its locals
            }

            // Sections for WebAssembly components
            InstanceSection(_) => { /* ... */ }
            CoreTypeSection(_) => { /* ... */ }
            ComponentInstanceSection(_) => { /* ... */ }
            ComponentAliasSection(_) => { /* ... */ }
            ComponentTypeSection(_) => { /* ... */ }
            ComponentCanonicalSection(_) => { /* ... */ }
            ComponentStartSection { .. } => { /* ... */ }
            ComponentImportSection(_) => { /* ... */ }
            ComponentExportSection(_) => { /* ... */ }

            ModuleSection { parser, .. }
            | ComponentSection { parser, .. } => {
                stack.push(cur.clone());
                cur = parser.clone();
            }

            CustomSection(_) => { /* ... */ }

            // most likely you'd return an error here
            UnknownSection { .. } => { /* ... */ }

            // Once we've reached the end of a parser we either resume
            // at the parent parser or we break out of the loop because
            // we're done.
            End(_) => {
                if let Some(parent_parser) = stack.pop() {
                    cur = parent_parser;
                } else {
                    break;
                }
            }
        }

        // once we're done processing the payload we can forget the
        // original.
        wasm_bytes.drain(..consumed);
    }
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm_mod).unwrap();
    for export in module.exports() {
        println!("{export:?}");
    }
    let linker = wasmtime::Linker::new(&engine);
    let mut store: wasmtime::Store<()> = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module).unwrap();
    println!("main: {:?}", instance.get_global(&mut store, "main").unwrap().get(&mut store));
}

#[cfg(not(feature="run"))]
fn run(_wasm_bytes: Vec<u8>) {
}

fn main() -> Result<(), ExitCode> {
    let args = Args::parse();
    let annot_arena = Arena::new();
    let mut interner = StringInterner::new();
    let mut globals: Globals = HashMap::new();

    // TODO: move this to the builtins themselves
    let add = interner.get_or_intern_static("add");
    let div = interner.get_or_intern_static("div");
    let mul = interner.get_or_intern_static("mul");
    let pi = interner.get_or_intern_static("pi");
    let sin = interner.get_or_intern_static("sin");
    let addone = interner.get_or_intern_static("addone");
    let reinterpi = interner.get_or_intern_static("reinterpi");
    let reinterpf = interner.get_or_intern_static("reinterpf");
    let cast = interner.get_or_intern_static("cast");
    globals.insert(add, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    globals.insert(div, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    globals.insert(mul, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    globals.insert(pi, typing::Type::Sample);
    globals.insert(sin, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)));
    globals.insert(addone, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)));
    globals.insert(reinterpi, typing::Type::Function(Box::new(typing::Type::Index), Box::new(typing::Type::Sample)));
    globals.insert(reinterpf, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Index)));
    globals.insert(cast, typing::Type::Function(Box::new(typing::Type::Index), Box::new(typing::Type::Sample)));

    let mut toplevel = TopLevel { arena: &annot_arena, interner, globals };

    let res = match args.cmd {
        Command::Parse { file, dump_to } => cmd_parse(&mut toplevel, file, dump_to),
        Command::Typecheck { file } => cmd_typecheck(&mut toplevel, file),
        Command::Interpret { file } => cmd_interpret(&mut toplevel, file),
        Command::Repl => cmd_repl(&mut toplevel),
        Command::Sample { out, file, length } => cmd_sample(&mut toplevel, file, length, out),
        Command::Compile { file, out } => cmd_compile(&mut toplevel, file, out),
    };

    match res {
        Ok(()) => { },
        Err(TopLevelError::TypeError(code, e)) => {
            println!("{}", e.pretty(&toplevel.interner, &code));
            return Err(1.into());
        },
        Err(err) => { println!("{:?}", err); return Err(2.into()); },
    }

    Ok(())
}
