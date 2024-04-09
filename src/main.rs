use core::fmt;
use std::error::Error;
use std::path::Path;
use std::{collections::HashMap, path::PathBuf, fs::File};
use std::io::Read;

use string_interner::StringInterner;

use typed_arena::Arena;

use clap::Parser as CliParser;

mod expr;
mod builtin;
mod parse;
mod interp;
mod typing;

use builtin::make_builtins;
use interp::get_samples;
use parse::Parser;

use crate::typing::Type;

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
    },
    /// Type-check the given program.
    TypeCheck {
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
}

fn read_file(name: Option<&Path>) -> std::io::Result<String> {
    let mut s = String::new();
    match name {
        Some(path) => { File::open(path)?.read_to_string(&mut s)?; },
        None => { std::io::stdin().read_to_string(&mut s)?; }
    }
    Ok(s)
}

struct TopLevel<'a, 'b> {
    parser: Parser<'a, 'b>,
    ctx: typing::Ctx,
}

#[derive(Debug)]
enum TopLevelError<'a> {
    IoError(std::io::Error),
    ParseError(String, parse::FullParseError),
    TypeError(String, typing::TypeError<'a, tree_sitter::Range>),
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

fn cmd_parse<'a>(toplevel: &mut TopLevel<'_, 'a>, file: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    match toplevel.parser.parse(&code) {
        Ok(expr) => {
            println!("{}", expr.pretty(toplevel.parser.interner));
        },
        Err(parse::FullParseError { tree, error }) => {
            eprintln!("{:?}", error);
            tree.print_dot_graph(&std::io::stdout());
        },
    }
    Ok(())
}

fn cmd_typecheck<'a>(toplevel: &mut TopLevel<'_, 'a>, file: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let expr = match toplevel.parser.parse(&code) {
        Ok(expr) => expr,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let expr = toplevel.parser.arena.alloc(expr);
    let ty = typing::synthesize(&toplevel.ctx, expr).map_err(|e| TopLevelError::TypeError(code, e))?;
    println!("synthesized type: {}", ty);
    Ok(())
}

fn cmd_interpret<'a>(toplevel: &mut TopLevel<'_, 'a>, file: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let expr = match toplevel.parser.parse(&code) {
        Ok(expr) => expr,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let expr = toplevel.parser.arena.alloc(expr);
    let ty = typing::synthesize(&toplevel.ctx, expr).map_err(|e| TopLevelError::TypeError(code, e))?;
    println!("synthesized type: {}", ty);

    let arena = Arena::new();
    let expr_unannotated = expr.map_ext(&arena, &(|_| ()));
    let builtins = make_builtins(&mut toplevel.parser.interner);
    let interp_ctx = interp::InterpretationContext { builtins: &builtins, env: HashMap::new() };

    match interp::interp(&interp_ctx, &expr_unannotated) {
        Ok(v) => println!("{:?}", v),
        Err(e) => return Err(TopLevelError::InterpError(e.into())),
    }

    Ok(())
}

fn repl_one<'a>(toplevel: &mut TopLevel<'_, 'a>, interp_ctx: &interp::InterpretationContext<'a, '_>, code: String) -> TopLevelResult<'a, ()> {
    let expr = match toplevel.parser.parse(&code) {
        Ok(expr) => expr,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let expr = toplevel.parser.arena.alloc(expr);
    let ty = typing::synthesize(&toplevel.ctx, expr).map_err(|e| TopLevelError::TypeError(code, e))?;
    println!("synthesized type: {}", ty);

    let arena = Arena::new();
    let expr_unannotated = expr.map_ext(&arena, &(|_| ()));

    match interp::interp(&interp_ctx, &expr_unannotated) {
        Ok(v) => println!("{:?}", v),
        Err(e) => return Err(TopLevelError::InterpError(e.into())),
    }

    Ok(())
}

fn cmd_repl<'a>(toplevel: &mut TopLevel<'_, 'a>) -> TopLevelResult<'a, ()> {
    let builtins = make_builtins(&mut toplevel.parser.interner);
    let interp_ctx = interp::InterpretationContext { builtins: &builtins, env: HashMap::new() };

    for line in std::io::stdin().lines() {
        match repl_one(toplevel, &interp_ctx, line?) {
            Ok(()) => { },
            Err(err @ TopLevelError::IoError(_)) => { return Err(err); },
            Err(err) => { println!("{:?}", err); },
        }
    }

    Ok(())
}

fn cmd_sample<'a>(toplevel: &mut TopLevel<'_, 'a>, file: Option<PathBuf>, length: usize, out: PathBuf) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let expr = match toplevel.parser.parse(&code) {
        Ok(expr) => expr,
        Err(e) => { return Err(TopLevelError::ParseError(code, e)); }
    };
    let expr = toplevel.parser.arena.alloc(expr);

    match typing::synthesize(&toplevel.ctx, expr).map_err(|e| TopLevelError::TypeError(code, e))? {
        Type::Stream(ty) if *ty == Type::Sample => { },
        ty => { return Err(TopLevelError::CannotSample(ty)); },
    }

    let arena = Arena::new();
    let expr_unannotated = expr.map_ext(&arena, &(|_| ()));
    let builtins = make_builtins(&mut toplevel.parser.interner);
    let mut samples = [0.0].repeat(48 * length);

    match get_samples(&builtins, &expr_unannotated, &mut samples[..]) {
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

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let annot_arena = Arena::new();
    let mut interner = StringInterner::new();
    let mut ctx: typing::Ctx = HashMap::new();

    // TODO: move this to the builtins themselves
    let add = interner.get_or_intern_static("add");
    let div = interner.get_or_intern_static("div");
    let mul = interner.get_or_intern_static("mul");
    let pi = interner.get_or_intern_static("pi");
    let sin = interner.get_or_intern_static("sin");
    ctx.insert(add, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    ctx.insert(div, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    ctx.insert(mul, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    ctx.insert(pi, typing::Type::Sample);
    ctx.insert(sin, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)));

    let parser = Parser::new(&mut interner, &annot_arena);
    let mut toplevel = TopLevel { parser, ctx };

    let res = match args.cmd {
        Command::Parse { file } => cmd_parse(&mut toplevel, file),
        Command::TypeCheck { file } => cmd_typecheck(&mut toplevel, file),
        Command::Interpret { file } => cmd_interpret(&mut toplevel, file),
        Command::Repl => cmd_repl(&mut toplevel),
        Command::Sample { out, file, length } => cmd_sample(&mut toplevel, file, length, out),
    };

    match res {
        Ok(()) => { },
        Err(TopLevelError::TypeError(code, e)) => {
            println!("{}", e.pretty(toplevel.parser.interner, &code));
        },
        Err(err) => { println!("{:?}", err); },
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
