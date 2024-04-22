use core::fmt;
use std::error::Error;
use std::path::Path;
use std::process::ExitCode;
use std::{collections::HashMap, path::PathBuf, fs::File};
use std::io::Read;

use expr::{Expr, Symbol};
use num::One;
use string_interner::{StringInterner, DefaultStringInterner};

use typed_arena::Arena;

use clap::Parser as CliParser;

mod expr;
mod builtin;
mod parse;
mod interp;
mod typing;
mod ir0;
mod ir1;
mod util;

use builtin::make_builtins;
use interp::get_samples;
use parse::Parser;
use typing::{Globals, Typechecker, Ctx};

use crate::typing::{Type, TopLevelTypeError, Kind, Clock};

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

    fn make_typechecker<'b>(&'b mut self) -> Typechecker<'b> {
        Typechecker { globals: &self.globals, interner: &mut self.interner }
    }
}

#[derive(Debug)]
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

/*
impl<'a> From<parse::FullParseError> for TopLevelError<'a> {
    fn from(err: parse::FullParseError) -> TopLevelError<'a> {
        TopLevelError::ParseError(err)
    }
}

impl<'a> From<typing::TypeError<'a, tree_sitter::Range>> for TopLevelError<'a> {
    fn from(err: typing::TypeError<'a, tree_sitter::Range>) -> TopLevelError<'a> {
        TopLevelError::TypeError(err)
    }
}
*/

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
    toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let arena = Arena::new();
    let defs: HashMap<Symbol, &Expr<'_, ()>> = parsed_file.defs.iter().map(|def| (def.name, &*arena.alloc(def.body.map_ext(&arena, &(|_| ()))))).collect();
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
    let ty = toplevel.make_typechecker().synthesize(&Ctx::Empty, expr).map_err(|e| TopLevelError::TypeError(code, typing::FileTypeErrors { errs: vec![TopLevelTypeError::TypeError(empty_symbol, e)] } ))?;
    println!("synthesized type: {}", ty.pretty(&toplevel.interner));

    let arena = Arena::new();
    let expr_unannotated = expr.map_ext(&arena, &(|_| ()));

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
    toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let arena = Arena::new();
    let defs: HashMap<Symbol, &Expr<'_, ()>> = parsed_file.defs.iter().map(|def| (def.name, &*arena.alloc(def.body.map_ext(&arena, &(|_| ()))))).collect();
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

fn cmd_compile<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| TopLevelError::TypeError(code, e))?;

    let arena = Arena::new();
    let defs: HashMap<Symbol, &Expr<'_, ()>> = parsed_file.defs.iter().map(|def| (def.name, &*arena.alloc(def.body.map_ext(&arena, &(|_| ()))))).collect();

    let mut builtin_globals = HashMap::new();
    builtin_globals.insert(toplevel.interner.get_or_intern_static("add"), ir1::Global(0));
    builtin_globals.insert(toplevel.interner.get_or_intern_static("div"), ir1::Global(1));
    builtin_globals.insert(toplevel.interner.get_or_intern_static("mul"), ir1::Global(2));
    builtin_globals.insert(toplevel.interner.get_or_intern_static("pi"), ir1::Global(3));
    builtin_globals.insert(toplevel.interner.get_or_intern_static("sin"), ir1::Global(4));

    let expr_under_arena = Arena::new();
    let expr_ptr_arena = Arena::new();
    let expr_arena = ir1::ExprArena { arena: &expr_under_arena, ptr_arena: &expr_ptr_arena };
    let translator = ir1::Translator { builtins: builtin_globals, arena: &expr_arena };

    // TODO: compile everything, ofc
    let main = *defs.get(&parsed_file.defs.last().unwrap().name).unwrap();
    let expr_ir = translator.translate(std::rc::Rc::new(ir1::Ctx::Empty), main);
    println!("{expr_ir:?}");

    // let rewritten = translator.rewrite(&expr_ir);
    // println!("{rewritten:?}");

    let annotated = translator.annotate_used_vars(&expr_ir);
    println!("{:?}", annotated.0);

    Ok(())
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
    globals.insert(add, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    globals.insert(div, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    globals.insert(mul, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    globals.insert(pi, typing::Type::Sample);
    globals.insert(sin, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)));

    let mut toplevel = TopLevel { arena: &annot_arena, interner, globals };

    let res = match args.cmd {
        Command::Parse { file, dump_to } => cmd_parse(&mut toplevel, file, dump_to),
        Command::Typecheck { file } => cmd_typecheck(&mut toplevel, file),
        Command::Interpret { file } => cmd_interpret(&mut toplevel, file),
        Command::Repl => cmd_repl(&mut toplevel),
        Command::Sample { out, file, length } => cmd_sample(&mut toplevel, file, length, out),
        Command::Compile { file } => cmd_compile(&mut toplevel, file),
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

    /*
    // don't ask why i have to use a String here instead of &str
    let mut try_typing = |code: String| {
        let mut parser = Parser::new(&mut interner, &annot_arena);
        let expr = match parser.parse(&code[..]) {
            Ok(e) => e,
            Err(err) => {
                println!("parse error: {:?}", err);
                return;
            },
        };
        println!("{}", expr.pretty(&interner));
        match typing::synthesize(&ctx, &expr) {
            Ok(ty) => println!("synthesized type: {}", ty),
            Err(err) => println!("type error: {}", err.pretty(&interner, &code[..])),
        }
    };

    for line in std::io::stdin().lines() {
        let line = line?;
        try_typing(line);
        // println!("{}", line);
    }

    Ok(())
    */

    /*
    try_typing(r"(\x. x) : index -> unit");
    try_typing(r"(\x. x) : sample -> sample");
    try_typing(r"(\x. y) : sample -> sample");
    try_typing(r"((\x. x) : sample -> sample) 3");
    try_typing(r"((\x. x) : sample -> sample) 1.5");
    try_typing(r"1.5 2");
    try_typing(r"1.5");

    try_typing(r"(&s. !s) : sample");
    // try_typing(r"(&f. \x. ");
    try_typing(r"((&s. ((\x. x :: !s (add x 1.0)) : sample -> ~sample)) : sample -> ~sample) 0.0");

    let source_code = r"let pifourth = div pi 4.0 in ((&s. \x. sin x :: !s (add x pifourth)) : sample -> ~sample) 0.0";
    try_typing(source_code);

    try_typing(r"let foo = ((2.0, 3) : sample * index) in (let (x, y) = foo in x)");

    let source_code = r"let persamp = (div (mul 440.0 (mul 2.0 pi)) 48000.0) in ((&s. \x. sin x :: !s (add x persamp)) : sample -> ~sample) 0.0";
    */

    /*

    let expr = parser.parse(source_code).unwrap();
    println!("\n{}", expr.pretty(&interner));

    let arena = Arena::new();
    let expr_unannotated = expr.map_ext(&arena, &(|_| ()));

    // println!("{:?}", interp(&Env::new(), &expr_unannotated));

    let builtins = make_builtins(&mut interner);

    let mut samples = [0f32; 48000];
    let before = Instant::now();
    let result = get_samples(&builtins, &expr_unannotated, &mut samples[..]);
    let elapsed = before.elapsed();
    println!("\n{:?}", result);
    // println!("{samples:?}");
    println!("took {}ms", elapsed.as_millis());
    */
}
