use std::fs::{self, File};
use std::io::Read;
#[cfg(feature = "run")]
use std::io;

use clocky::toplevel::{compile, TopLevel};
#[cfg(feature = "run")]
use clocky::toplevel::run;
#[cfg(feature = "run")]
use hound::{Error::IoError, WavReader};
use typed_arena::Arena;

#[test]
fn test_accepts() {
    for test_file in fs::read_dir("tests/accept").unwrap() {
        let test_file_path = test_file.unwrap().path();
        // yes this weird construction is necessary, `ext != Some("rs")` will not work
        println!("found file {:?}", &test_file_path);
        if !matches!(test_file_path.extension(), Some(ext) if ext == "cky") {
            continue;
        }
        println!("testing file {:?}", &test_file_path);
        let mut code = String::new();
        File::open(&test_file_path).unwrap().read_to_string(&mut code).unwrap();
        let arena = Arena::new();
        let mut toplevel = TopLevel::new(&arena);
        let wasm_bytes = compile(&mut toplevel, code).unwrap();
        #[cfg(feature = "run")]
        {
            let wav_file = match WavReader::open(test_file_path.with_extension("wav")) {
                Ok(f) => f,
                Err(IoError(err)) if err.kind() == io::ErrorKind::NotFound => {
                    println!("...no wav file, not running the compilation result");
                    continue;
                },
                Err(err) => panic!("hound error: {}", err),
            };
            let ran_samples = run(&wasm_bytes, 48000 * 2);
            let expected_samples = wav_file.into_samples().collect::<Result<Vec<f32>, _>>().unwrap();
            assert_eq!(ran_samples, expected_samples);
        }
        drop(wasm_bytes);
    }
}
