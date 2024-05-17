pub mod expr;
pub mod builtin;
pub mod parse;
pub mod interp;
pub mod typing;
pub mod ir1;
pub mod ir2;
pub mod util;
pub mod wasm;
pub mod runtime;

#[cfg(target_arch = "wasm32")]
pub mod bindings;
