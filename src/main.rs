use std::path::Path;
use std::process::ExitCode;
use std::{path::PathBuf, fs::File};
use std::io::{Read, Write};

use typed_arena::Arena;

use clap::Parser as CliParser;

use clocky::parse;
use clocky::toplevel::{compile, TopLevel, TopLevelError, TopLevelResult};

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
    Compile {
        /// Code file to use
        file: Option<PathBuf>,

        /// Path to wasm module file to be written
        #[arg(short='o')]
        out: Option<PathBuf>,
    },
    Egglog {
        /// Code file to use
        file: Option<PathBuf>,
    },
    #[cfg(feature="run")]
    Sample {
        /// Code file to use
        file: PathBuf,

        /// Path to wav file to write to
        out: PathBuf,

        /// Length of time to sample for
        #[arg(short='l', default_value_t=10.0)]
        length: f32,
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

fn cmd_compile<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>, out: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let wasm_bytes = compile(toplevel, code)?;

    write_file(out.as_deref(), &wasm_bytes)?;

    Ok(())
}

fn cmd_egglog<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    clocky::toplevel::egglog(toplevel, code)?;

    Ok(())
}

#[cfg(feature="run")]
fn cmd_sample<'a>(toplevel: &mut TopLevel<'a>, file: PathBuf, out: PathBuf, length: f32) -> TopLevelResult<'a, ()> {
    let wasm_bytes = match file.extension() {
        Some(ext) if ext == "wasm" => {
            let mut buf = Vec::new();
            File::open(file)?.read_to_end(&mut buf)?;
            buf
        },
        _ => {
            let code = read_file(Some(&file))?;
            compile(toplevel, code)?
        },
    };

    let num_samples = (length * 48000.0) as usize;
    let samples = clocky::toplevel::run(&wasm_bytes, num_samples);

    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut hound_writer = hound::WavWriter::create(out, wav_spec).unwrap();

    for sample in samples {
        hound_writer.write_sample(sample).unwrap();
    }

    Ok(())
}

fn main() -> Result<(), ExitCode> {
    let args = Args::parse();

    let arena = Arena::new();
    let mut toplevel = TopLevel::new(&arena);

    let res = match args.cmd {
        Command::Parse { file, dump_to } => cmd_parse(&mut toplevel, file, dump_to),
        Command::Typecheck { file } => cmd_typecheck(&mut toplevel, file),
        Command::Compile { file, out } => cmd_compile(&mut toplevel, file, out),
        Command::Egglog { file } => cmd_egglog(&mut toplevel, file),
        #[cfg(feature = "run")]
        Command::Sample { file, out, length } => cmd_sample(&mut toplevel, file, out, length),
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
