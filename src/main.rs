use string_interner::StringInterner;

use typed_arena::Arena;

mod expr;
mod builtin;
mod parse;
mod interp;

use builtin::make_builtins;
use interp::get_samples;
use parse::Parser;

fn main() {
    let annot_arena = Arena::new();
    let mut interner = StringInterner::new();
    let mut parser = Parser::new(&mut interner, &annot_arena);

    let source_code = r"(&s. \x. sin x :: !s (add x (div pi 4.0))) 0.0";

    let expr = parser.parse(source_code).unwrap();
    println!("\n{}", expr.pretty(&interner));

    let arena = Arena::new();
    let expr_unannotated = expr.map_ext(&arena, &(|_| ()));

    // println!("{:?}", interp(&Env::new(), &expr_unannotated));

    let builtins = make_builtins(&mut interner);

    let mut samples = [0f32; 8];
    println!("\n{:?}", get_samples(&builtins, &expr_unannotated, &mut samples[..]));
    println!("{samples:?}");
}
