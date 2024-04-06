use std::{time::Instant, collections::HashMap};

use string_interner::StringInterner;

use typed_arena::Arena;

mod expr;
mod builtin;
mod parse;
mod interp;
mod typing;

use builtin::make_builtins;
use interp::get_samples;
use parse::Parser;

fn main() -> std::io::Result<()> {
    let annot_arena = Arena::new();
    let mut interner = StringInterner::new();
    let mut ctx: typing::Ctx = HashMap::new();
    let add = interner.get_or_intern_static("add");
    let div = interner.get_or_intern_static("div");
    let pi = interner.get_or_intern_static("pi");
    let sin = interner.get_or_intern_static("sin");
    ctx.insert(add, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    ctx.insert(div, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)))));
    ctx.insert(pi, typing::Type::Sample);
    ctx.insert(sin, typing::Type::Function(Box::new(typing::Type::Sample), Box::new(typing::Type::Sample)));

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
            Ok(ty) => println!("synthesized type: {:?}", ty),
            Err(err) => println!("type error: {}", err.pretty(&interner, &code[..])),
        }
    };

    for line in std::io::stdin().lines() {
        let line = line?;
        try_typing(line);
        // println!("{}", line);
    }

    Ok(())

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
