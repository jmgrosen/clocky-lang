use std::collections::HashMap;
use std::fmt;

use string_interner::{DefaultStringInterner, DefaultSymbol};
use typed_arena::Arena;

use crate::builtin::Builtin;
use crate::typing::{Clock, Type, PrettyClock};

pub type Symbol = DefaultSymbol;

#[derive(Debug, Clone)]
pub enum Value<'a> {
    Unit,
    Sample(f32),
    Index(usize),
    Pair(Box<Value<'a>>, Box<Value<'a>>),
    InL(Box<Value<'a>>),
    InR(Box<Value<'a>>),
    Gen(Env<'a>, Box<Value<'a>>, &'a Expr<'a, ()>),
    Closure(Env<'a>, Symbol, &'a Expr<'a, ()>),
    Suspend(Env<'a>, &'a Expr<'a, ()>),
    BuiltinPartial(Builtin, Box<[Value<'a>]>),
    Array(Box<[Value<'a>]>),
}

// TODO: use a better type here... eventually we should resolve symbols and just use de bruijn offsets or similar...
pub type Env<'a> = HashMap<Symbol, Value<'a>>;
// type Env<'a> = imbl::HashMap<Symbol, Value<'a>>;

#[derive(Debug, Clone)]
pub enum Expr<'a, R> {
    Var(R, Symbol),
    Val(R, Value<'a>),
    Annotate(R, &'a Expr<'a, R>, Type),
    Lam(R, Symbol, &'a Expr<'a, R>),
    App(R, &'a Expr<'a, R>, &'a Expr<'a, R>),
    Force(R, &'a Expr<'a, R>),
    Lob(R, Clock, Symbol, &'a Expr<'a, R>),
    Gen(R, &'a Expr<'a, R>, &'a Expr<'a, R>),
    LetIn(R, Symbol, &'a Expr<'a, R>, &'a Expr<'a, R>),
    Pair(R, &'a Expr<'a, R>, &'a Expr<'a, R>),
    UnPair(R, Symbol, Symbol, &'a Expr<'a, R>, &'a Expr<'a, R>),
    InL(R, &'a Expr<'a, R>),
    InR(R, &'a Expr<'a, R>),
    Case(R, &'a Expr<'a, R>, Symbol, &'a Expr<'a, R>, Symbol, &'a Expr<'a, R>),
    Array(R, Box<[&'a Expr<'a, R>]>),
    UnGen(R, &'a Expr<'a, R>),
}

impl<'a, R> Expr<'a, R> {
    pub fn map_ext<'b, U>(&self, arena: &'b Arena<Expr<'b, U>>, f: &dyn Fn(&R) -> U) -> Expr<'b, U> where 'a: 'b {
        match *self {
            Expr::Var(ref r, s) => Expr::Var(f(r), s),
            Expr::Val(ref r, ref v) => Expr::Val(f(r), v.clone()),
            Expr::Annotate(ref r, e, ref ty) => Expr::Annotate(f(r), arena.alloc(e.map_ext(arena, f)), ty.clone()),
            Expr::Lam(ref r, s, ref e) => Expr::Lam(f(r), s, arena.alloc(e.map_ext(arena, f))),
            Expr::App(ref r, ref e1, ref e2) => Expr::App(f(r), arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::Force(ref r, ref e) => Expr::Force(f(r), arena.alloc(e.map_ext(arena, f))),
            Expr::Lob(ref r, clock, s, ref e) => Expr::Lob(f(r), clock, s, arena.alloc(e.map_ext(arena, f))),
            Expr::Gen(ref r, ref e1, ref e2) => Expr::Gen(f(r), arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::LetIn(ref r, s, ref e1, ref e2) => Expr::LetIn(f(r), s, arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::Pair(ref r, e1, e2) => Expr::Pair(f(r), arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::UnPair(ref r, s1, s2, e1, e2) => Expr::UnPair(f(r), s1, s2, arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::InL(ref r, e) => Expr::InL(f(r), arena.alloc(e.map_ext(arena, f))),
            Expr::InR(ref r, e) => Expr::InR(f(r), arena.alloc(e.map_ext(arena, f))),
            Expr::Case(ref r, e0, s1, e1, s2, e2) => Expr::Case(f(r), arena.alloc(e0.map_ext(arena, f)), s1, arena.alloc(e1.map_ext(arena, f)), s2, arena.alloc(e2.map_ext(arena, f))),
            Expr::Array(ref r, ref es) => Expr::Array(f(r), es.iter().map(|e| &*arena.alloc(e.map_ext(arena, f))).collect::<Vec<_>>().into()),
            Expr::UnGen(ref r, ref e) => Expr::UnGen(f(r), arena.alloc(e.map_ext(arena, f))),
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

    fn for_clock(&self, clock: &'a Clock) -> PrettyClock<'a> {
        clock.pretty(self.interner)
    }

    fn name(&self, s: Symbol) -> &'a str {
        self.interner.resolve(s).expect("encountered an symbol not corresponding to an identifier while pretty printing an expression")
    }
}

impl<'a, 'b, R> fmt::Display for PrettyExpr<'a, 'b, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.expr {
            Expr::Var(_, x) =>
                write!(f, "Var({})", self.name(x)),
            Expr::Val(_, ref v) =>
                write!(f, "{:?}", v),
            Expr::Annotate(_, e, ref ty) =>
                write!(f, "Annotate({}, {:?})", self.for_expr(e), ty),
            Expr::App(_, ref e1, ref e2) =>
                write!(f, "App({}, {})", self.for_expr(e1), self.for_expr(e2)),
            Expr::Lam(_, x, ref e) =>
                write!(f, "Lam({}, {})", self.name(x), self.for_expr(e)),
            Expr::Force(_, ref e) =>
                write!(f, "Force({})", self.for_expr(e)),
            Expr::Lob(_, ref clock, x, ref e) =>
                write!(f, "Lob({}, {}, {})", self.for_clock(clock), self.name(x), self.for_expr(e)),
            Expr::Gen(_, ref eh, ref et) =>
                write!(f, "Gen({}, {})", self.for_expr(eh), self.for_expr(et)),
            Expr::LetIn(_, x, e1, e2) =>
                write!(f, "Let({}, {}, {})", self.name(x), self.for_expr(e1), self.for_expr(e2)),
            Expr::Pair(_, e1, e2) =>
                write!(f, "Pair({}, {})", self.for_expr(e1), self.for_expr(e2)),
            Expr::UnPair(_, x1, x2, e1, e2) =>
                write!(f, "UnPair({}, {}, {}, {})", self.name(x1), self.name(x2), self.for_expr(e1), self.for_expr(e2)),
            Expr::InL(_, e) =>
                write!(f, "InL({})", self.for_expr(e)),
            Expr::InR(_, e) =>
                write!(f, "InR({})", self.for_expr(e)),
            Expr::Case(_, e0, x1, e1, x2, e2) =>
                write!(f, "Case({}, {}, {}, {}, {})", self.for_expr(e0), self.name(x1), self.for_expr(e1), self.name(x2), self.for_expr(e2)),
            Expr::Array(_, ref es) => {
                write!(f, "Array(")?;
                for (i, e) in es.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.for_expr(e))?;
                }
                write!(f, ")")
            },
            Expr::UnGen(_, ref e) =>
                write!(f, "UnGen({})", self.for_expr(e)),
        }
    }
}
