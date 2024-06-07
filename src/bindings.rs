use typed_arena::Arena;

use wasm_bindgen::prelude::*;

use crate::toplevel::{self, TopLevel};

#[wasm_bindgen]
pub fn compile(code: String) -> Result<Vec<u8>, String> {
    let arena = Arena::new();
    let mut toplevel = TopLevel::new(&arena);
    match toplevel::compile(&mut toplevel, code) {
        Ok(wasm_bytes) =>
            Ok(wasm_bytes),
        Err(err) =>
            Err(format!("{:?}", err)),
    }
}
