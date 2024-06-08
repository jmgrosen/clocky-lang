use std::path::Path;
use std::process::ExitCode;
use std::{path::PathBuf, fs::File};
use std::io::{Read, Write};

use byteorder::{LittleEndian, ReadBytesExt};

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
    #[cfg(feature="run")]
    Sample {
        /// Code file to use
        file: Option<PathBuf>,

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

#[cfg(feature="run")]
fn cmd_sample<'a>(toplevel: &mut TopLevel<'a>, file: Option<PathBuf>, out: PathBuf, length: f32) -> TopLevelResult<'a, ()> {
    let code = read_file(file.as_deref())?;
    let wasm_bytes = compile(toplevel, code)?;

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm_bytes).unwrap();
    for export in module.exports() {
        println!("{export:?}");
    }
    let linker = wasmtime::Linker::new(&engine);
    let mut store: wasmtime::Store<()> = wasmtime::Store::new(&engine, ());
    let instance = linker.instantiate(&mut store, &module).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();

    let alloc = instance.get_typed_func::<u32, u32>(&mut store, "alloc").unwrap();
    let sample_scheduler = instance.get_typed_func::<(u32, u32, u32), u32>(&mut store, "sample_scheduler").unwrap();

    let samples_ptr = alloc.call(&mut store, 4 * 128).unwrap();
    let mut main = instance.get_global(&mut store, "main").unwrap().get(&mut store).unwrap_i32() as u32;

    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut hound_writer = hound::WavWriter::create(out, wav_spec).unwrap();

    const CHUNK_SIZE: u32 = 128;
    let num_chunks = (length * 48000.0 / (CHUNK_SIZE as f32)) as usize;
    for _ in 0..num_chunks {
        main = sample_scheduler.call(&mut store, (main, CHUNK_SIZE, samples_ptr)).unwrap();
        let mut samples_buf = [0u8; CHUNK_SIZE as usize*4];
        memory.read(&mut store, samples_ptr as usize, &mut samples_buf).unwrap();
        let mut remaining_samples = &samples_buf[..];
        while remaining_samples.len() > 0 {
            hound_writer.write_sample(remaining_samples.read_f32::<LittleEndian>().unwrap()).unwrap();
        }
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
