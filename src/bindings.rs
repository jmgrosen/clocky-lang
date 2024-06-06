use core::fmt;
use std::error::Error;
use std::path::Path;
use std::process::ExitCode;
use std::{collections::HashMap, path::PathBuf, fs::File};
use std::io::{Read, Write};

use crate::expr::{Expr, Symbol};
use num::One;
use string_interner::{StringInterner, DefaultStringInterner};

use typed_arena::Arena;

use wasm_bindgen::prelude::*;

use clap::Parser as CliParser;

use crate::builtin::make_builtins;
use crate::interp::get_samples;
use crate::parse::{self, Parser};
use crate::typing::{self, Globals, Typechecker, Ctx};
use crate::{ir1, ir2, interp, wasm, util};

use crate::typing::{Type, TopLevelTypeError, Kind, Clock};

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
        Typechecker { arena: self.arena, globals: &mut self.globals, interner: &mut self.interner }
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

#[wasm_bindgen]
pub fn compile(code: String) -> Result<Vec<u8>, String> {
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
    let since_tick = interner.get_or_intern_static("since_tick");
    let c = interner.get_or_intern_static("c");
    globals.insert(since_tick, typing::Type::Forall(c, Kind::Clock, Box::new(typing::Type::Stream(typing::Clock::from_var(c), Box::new(typing::Type::Sample)))));

    let mut toplevel = TopLevel { arena: &annot_arena, interner, globals };

    let parsed_file = match toplevel.make_parser().parse_file(&code) {
        Ok(parsed_file) => parsed_file,
        Err(e) => { return Err(format!("{:?}", TopLevelError::ParseError(code, e))); }
    };
    let elabbed_file = toplevel.make_typechecker().check_file(&parsed_file).map_err(|e| format!("{:?}", TopLevelError::TypeError(code, e)))?;

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
    let translator = ir1::Translator { globals: builtin_globals, global_clocks: HashMap::new(), arena: &expr_arena };

    
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
                expr2_arena.alloc(ir2::Expr::Var(ir1::DebruijnIndex(i)))
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
