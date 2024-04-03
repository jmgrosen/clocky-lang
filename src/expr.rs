use std::collections::HashMap;
use std::fmt;

use string_interner::{DefaultStringInterner, DefaultSymbol};
use typed_arena::Arena;

use crate::builtin::Builtin;

pub type Symbol = DefaultSymbol;

#[derive(Debug, Clone)]
pub enum Value<'a> {
    Sample(f32),
    Index(usize),
    Gen(Env<'a>, Box<Value<'a>>, &'a Expr<'a, ()>),
    Closure(Env<'a>, Symbol, &'a Expr<'a, ()>),
    Suspend(Env<'a>, &'a Expr<'a, ()>),
    BuiltinPartial(Builtin, Box<[Value<'a>]>),
}

// TODO: use a better type here... eventually we should resolve symbols and just use de bruijn offsets or similar...
pub type Env<'a> = HashMap<Symbol, Value<'a>>;
// type Env<'a> = imbl::HashMap<Symbol, Value<'a>>;

#[derive(Debug, Clone)]
pub enum Expr<'a, R> {
    Var(R, Symbol),
    Val(R, Value<'a>),
    Lam(R, Symbol, &'a Expr<'a, R>),
    App(R, &'a Expr<'a, R>, &'a Expr<'a, R>),
    Force(R, &'a Expr<'a, R>),
    Lob(R, Symbol, &'a Expr<'a, R>),
    Gen(R, &'a Expr<'a, R>, &'a Expr<'a, R>),
    LetIn(R, Symbol, &'a Expr<'a, R>, &'a Expr<'a, R>),
}

impl<'a, R> Expr<'a, R> {
    pub fn map_ext<'b, U>(&self, arena: &'b Arena<Expr<'b, U>>, f: &dyn Fn(&R) -> U) -> Expr<'b, U> where 'a: 'b {
        match *self {
            Expr::Var(ref r, s) => Expr::Var(f(r), s),
            Expr::Val(ref r, ref v) => Expr::Val(f(r), v.clone()),
            Expr::Lam(ref r, s, ref e) => Expr::Lam(f(r), s, arena.alloc(e.map_ext(arena, f))),
            Expr::App(ref r, ref e1, ref e2) => Expr::App(f(r), arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::Force(ref r, ref e) => Expr::Force(f(r), arena.alloc(e.map_ext(arena, f))),
            Expr::Lob(ref r, s, ref e) => Expr::Lob(f(r), s, arena.alloc(e.map_ext(arena, f))),
            Expr::Gen(ref r, ref e1, ref e2) => Expr::Gen(f(r), arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::LetIn(ref r, s, ref e1, ref e2) => Expr::LetIn(f(r), s, arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
        }
    }

    pub fn pretty<'b>(&'b self, interner: &'b DefaultStringInterner) -> PrettyExpr<'b, 'a, R> {
        PrettyExpr { interner, expr: self }
    }
}

pub struct PrettyExpr<'a, 'b, R> {
    interner: &'a DefaultStringInterner,
    expr: &'a Expr<'b, R>,
}

impl<'a, 'b, R> PrettyExpr<'a, 'b, R> {
    fn for_expr(&self, other_expr: &'a Expr<'b, R>) -> PrettyExpr<'a, 'b, R> {
        PrettyExpr { interner: self.interner, expr: other_expr }
    }
}

impl<'a, 'b, R> fmt::Display for PrettyExpr<'a, 'b, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.expr {
            Expr::Var(_, x) =>
                write!(f, "Var({})", self.interner.resolve(x).unwrap()),
            Expr::Val(_, ref v) =>
                write!(f, "{:?}", v),
            Expr::App(_, ref e1, ref e2) =>
                write!(f, "App({}, {})", self.for_expr(e1), self.for_expr(e2))
            ,
            Expr::Lam(_, x, ref e) => {
                let x_str = self.interner.resolve(x).unwrap();
                write!(f, "Lam({}, {})", x_str, self.for_expr(e))
            },
            Expr::Force(_, ref e) =>
                write!(f, "Force({})", self.for_expr(e)),
            Expr::Lob(_, x, ref e) => {
                let x_str = self.interner.resolve(x).unwrap();
                write!(f, "Lob({}, {})", x_str, self.for_expr(e))
            },
            Expr::Gen(_, ref eh, ref et) =>
                write!(f, "Gen({}, {})", self.for_expr(eh), self.for_expr(et)),
            Expr::LetIn(_, x, e1, e2) =>
                write!(f, "Let({}, {}, {})", self.interner.resolve(x).unwrap(), self.for_expr(e1), self.for_expr(e2)),
        }
    }
}
