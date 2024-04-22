use std::{rc::Rc, collections::{HashMap, HashSet}, mem::MaybeUninit, fmt};

use typed_arena::Arena;

use crate::expr::{Expr as HExpr, Symbol, Value as HValue};
// use crate::util::parenthesize;

// okay, for real, what am i really getting here over just using u32...
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DebruijnIndex(u32);

impl fmt::Debug for DebruijnIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl DebruijnIndex {
    const HERE: DebruijnIndex = DebruijnIndex(0);

    fn shifted(self) -> DebruijnIndex {
        DebruijnIndex(self.0 + 1)
    }

    fn shifted_by(self, by: u32) -> DebruijnIndex {
        DebruijnIndex(self.0 + by)
    }

    fn shifted_by_signed(self, by: i32) -> DebruijnIndex {
        if let Some(new) = self.0.checked_add_signed(by) {
            DebruijnIndex(new)
        } else {
            panic!("debruijn underflow")
        }
    }

    fn is_within(&self, depth: u32) -> bool {
        self.0 < depth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Global(pub u32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Unit,
    Sample(f32),
    Index(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Value::Unit => write!(f, "()"),
            Value::Sample(x) => write!(f, "{}", x),
            Value::Index(i) => write!(f, "{}", i),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Con {
    Stream,
    Array,
    Pair,
    InL,
    InR,
}

impl Con {
    // returns None if variable arity
    fn arity(&self) -> Option<u8> {
        match *self {
            Con::Stream => Some(2),
            Con::Array => None,
            Con::Pair => Some(2),
            Con::InL => Some(1),
            Con::InR => Some(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Pi,
    Fst,
    Snd,
    UnGen,
    Force,
}

impl Op {
    fn arity(&self) -> u8 {
        match *self {
            Op::Add => 2,
            Op::Sub => 2,
            Op::Mul => 2,
            Op::Div => 2,
            Op::Sin => 1,
            Op::Cos => 1,
            Op::Pi => 0,
            Op::Fst => 1,
            Op::Snd => 1,
            Op::UnGen => 1,
            Op::Force => 1,
        }
    }
}

type VarSet = HashSet<DebruijnIndex>;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    Var(DebruijnIndex),
    Val(Value),
    Glob(Global),
    Lam(Option<VarSet>, u32, &'a Expr<'a>),
    App(&'a Expr<'a>, &'a [&'a Expr<'a>]),
    Unbox(&'a Expr<'a>),
    Box(Option<VarSet>, &'a Expr<'a>),
    Lob(Option<VarSet>, &'a Expr<'a>),
    LetIn(&'a Expr<'a>, &'a Expr<'a>),
    Case(&'a Expr<'a>, &'a Expr<'a>, &'a Expr<'a>),
    Con(Con, &'a [&'a Expr<'a>]),
    Op(Op, &'a [&'a Expr<'a>]),
    // These two are initially treated separately from Force and
    // Suspend because after the initial translation, we have to do
    // the rewriting step of pulling Advances out of Delays and
    // Lams... and maybe we will want to handle allocation/GC
    // differently for them? idk
    Delay(Option<VarSet>, &'a Expr<'a>),
    Adv(&'a Expr<'a>),
}

/*
impl<'a> fmt::Display for Expr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr::*;
        // okay this time let's do something hacky and use the
        // "precision" specified in the formatter as precedence
        let prec = f.precision().unwrap_or(0);
        match *self {
            Var(i) =>
                write!(f, "#{:?}", i),
            Val(v) =>
                write!(f, "{}", v),
            Glob(Global(g)) =>
                write!(f, "@{}", g),
            Lam(arity, e) =>
                parenthesize(f, prec > 0, |f| {
                    write!(f, "\\");
        }
    }
}
*/

pub struct ExprArena<'a> {
    pub arena: &'a Arena<Expr<'a>>,
    pub ptr_arena: &'a Arena<&'a Expr<'a>>,
}

impl<'a> ExprArena<'a> {
    fn alloc(&self, e: Expr<'a>) -> &'a Expr<'a> {
        self.arena.alloc(e)
    }

    fn alloc_slice(&self, it: impl IntoIterator<Item=&'a Expr<'a>>) -> &'a [&'a Expr<'a>] {
        self.ptr_arena.alloc_extend(it)
    }

    /// A form of alloc_slice that is reentrant: you can use the arena
    /// in the iterator. It requires the iterator to return a correct
    /// upper bound size hint.
    fn alloc_slice_r(&self, it: impl IntoIterator<Item=&'a Expr<'a>>) -> &'a [&'a Expr<'a>] {
        // TODO: is this safe? my gut says, "not technically but
        // probably fine in practice." this will end up with
        // uninitialized references within ptr_arena, which rust
        // sounds like it *really* does not like, but it's
        // prooooooobably fine, right? perhaps this could be fixed by
        // changing ptr_arena to instead allocate MaybeUninit<&'a
        // Expr<'a>>?
        let it = it.into_iter();
        let (_, Some(bound)) = it.size_hint() else {
            panic!("alloc_slice_r called with iterator that does not return an upper bound")
        };

        unsafe {
            let uninit = self.ptr_arena.alloc_uninitialized(bound);
            // don't use .enumerate() bc we need the count afterwards
            let mut i = 0;
            for e in it {
                uninit[i].write(e);
                i += 1;
            }

            let valid_portion = &uninit[..i];
            &*(valid_portion as *const [MaybeUninit<&'a Expr<'a>>] as *const [&'a Expr<'a>])
        }
    }

    fn alloc_slice_alloc(&self, it: impl IntoIterator<Item=Expr<'a>>) -> &'a [&'a Expr<'a>] {
        self.alloc_slice(it.into_iter().map(|e| self.alloc(e)))
    }

    fn alloc_slice_maybe(&self, it: impl IntoIterator<Item=Option<&'a Expr<'a>>>) -> Option<&'a [&'a Expr<'a>]> {
        let mut success = true;
        let slice = self.ptr_arena.alloc_extend(it.into_iter().map_while(|e| { success = e.is_some(); e }));
        if success { Some(slice) } else { None }
    }

    /// A form of alloc_slice that is reentrant: you can use the arena
    /// in the iterator. It requires the iterator to return a correct
    /// upper bound size hint.
    fn alloc_slice_maybe_r(&self, it: impl IntoIterator<Item=Option<&'a Expr<'a>>>) -> Option<&'a [&'a Expr<'a>]> {
        // TODO: is this safe? my gut says, "not technically but
        // probably fine in practice." this will end up with
        // uninitialized references within ptr_arena, which rust
        // sounds like it *really* does not like, but it's
        // prooooooobably fine, right? perhaps this could be fixed by
        // changing ptr_arena to instead allocate MaybeUninit<&'a
        // Expr<'a>>?
        let it = it.into_iter();
        let (_, Some(bound)) = it.size_hint() else {
            panic!("alloc_slice_r called with iterator that does not return an upper bound")
        };

        unsafe {
            let uninit = self.ptr_arena.alloc_uninitialized(bound);
            // don't use .enumerate() bc we need the count afterwards
            let mut i = 0;
            for oe in it {
                if let Some(e) = oe {
                    uninit[i].write(e);
                    i += 1;
                } else {
                    return None;
                }
            }

            let valid_portion = &uninit[..i];
            Some(&*(valid_portion as *const [MaybeUninit<&'a Expr<'a>>] as *const [&'a Expr<'a>]))
        }
    }

    fn alloc_slice_maybe_alloc(&self, it: impl IntoIterator<Item=Option<Expr<'a>>>) -> Option<&'a [&'a Expr<'a>]> {
        self.alloc_slice_maybe(it.into_iter().map(|oe| oe.map(|e| self.alloc(e))))
    }
}

impl<'a> Expr<'a> {
    fn shifted_by<'b>(&self, by: u32, depth: u32, arena: &'b ExprArena<'b>) -> Expr<'b> {
        use Expr::*;
        match *self {
            Var(i) => if i.is_within(depth) { Var(i) } else { Var(i.shifted_by(by)) },
            Val(v) => Val(v),
            Glob(g) => Glob(g),
            Lam(ref vs, arity, e) => Lam(vs.clone(), arity, arena.alloc(e.shifted_by(by, depth + arity, arena))),
            App(e0, es) => App(
                arena.alloc(e0.shifted_by(by, depth, arena)),
                arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by(by, depth, arena))))
            ),
            Unbox(e) => Unbox(arena.alloc(e.shifted_by(by, depth, arena))),
            Box(ref vs, e) => Box(vs.clone(), arena.alloc(e.shifted_by(by, depth, arena))),
            Lob(ref vs, e) => Lob(vs.clone(), arena.alloc(e.shifted_by(by, depth + 1, arena))),
            LetIn(e1, e2) => LetIn(
                arena.alloc(e1.shifted_by(by, depth, arena)),
                arena.alloc(e2.shifted_by(by, depth + 1, arena))
            ),
            Case(e0, e1, e2) => Case(
                arena.alloc(e0.shifted_by(by, depth, arena)),
                arena.alloc(e1.shifted_by(by, depth + 1, arena)),
                arena.alloc(e2.shifted_by(by, depth + 1, arena))
            ),
            Con(con, es) => Con(con, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by(by, depth, arena))))),
            Op(op, es) => Op(op, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by(by, depth, arena))))),
            Delay(ref vs, e) => Delay(vs.clone(), arena.alloc(e.shifted_by(by, depth, arena))),
            Adv(e) => Adv(arena.alloc(e.shifted_by(by, depth, arena))),
        }
    }

    fn shifted_by_signed<'b>(&self, by: i32, depth: u32, arena: &'b ExprArena<'b>) -> Expr<'b> {
        use Expr::*;
        match *self {
            Var(i) => if i.is_within(depth) { Var(i) } else { Var(i.shifted_by_signed(by)) },
            Val(v) => Val(v),
            Glob(g) => Glob(g),
            Lam(ref vs, arity, e) => Lam(vs.clone(), arity, arena.alloc(e.shifted_by_signed(by, depth + arity, arena))),
            App(e0, es) => App(
                arena.alloc(e0.shifted_by_signed(by, depth, arena)),
                arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by_signed(by, depth, arena))))
            ),
            Unbox(e) => Unbox(arena.alloc(e.shifted_by_signed(by, depth, arena))),
            Box(ref vs, e) => Box(vs.clone(), arena.alloc(e.shifted_by_signed(by, depth, arena))),
            Lob(ref vs, e) => Lob(vs.clone(), arena.alloc(e.shifted_by_signed(by, depth + 1, arena))),
            LetIn(e1, e2) => LetIn(
                arena.alloc(e1.shifted_by_signed(by, depth, arena)),
                arena.alloc(e2.shifted_by_signed(by, depth + 1, arena))
            ),
            Case(e0, e1, e2) => Case(
                arena.alloc(e0.shifted_by_signed(by, depth, arena)),
                arena.alloc(e1.shifted_by_signed(by, depth + 1, arena)),
                arena.alloc(e2.shifted_by_signed(by, depth + 1, arena))
            ),
            Con(con, es) => Con(con, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by_signed(by, depth, arena))))),
            Op(op, es) => Op(op, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by_signed(by, depth, arena))))),
            Delay(ref vs, e) => Delay(vs.clone(), arena.alloc(e.shifted_by_signed(by, depth, arena))),
            Adv(e) => Adv(arena.alloc(e.shifted_by_signed(by, depth, arena))),
        }
    }
}
            

pub struct Translator<'a> {
    pub builtins: HashMap<Symbol, Global>,
    pub arena: &'a ExprArena<'a>,
}

pub enum Ctx {
    Empty,
    Var(Symbol, Rc<Ctx>),
    Silent(Rc<Ctx>),
}

impl Ctx {
    fn lookup(&self, x: Symbol) -> Option<DebruijnIndex> {
        match *self {
            Ctx::Empty =>
                None,
            Ctx::Var(y, _) if x == y =>
                Some(DebruijnIndex::HERE),
            Ctx::Var(_, ref next) =>
                next.lookup(x).map(|i| i.shifted()),
            Ctx::Silent(ref next) =>
                next.lookup(x).map(|i| i.shifted()),
        }
    }
}

impl<'a> Translator<'a> {
    fn alloc(&self, e: Expr<'a>) -> &'a Expr<'a> {
        self.arena.alloc(e)
    }

    fn alloc_slice(&self, it: impl IntoIterator<Item=&'a Expr<'a>>) -> &'a [&'a Expr<'a>] {
        self.arena.alloc_slice(it)
    }

    pub fn translate<'b, R>(&self, ctx: Rc<Ctx>, expr: &'b HExpr<'b, R>) -> Expr<'a> {
        match *expr {
            HExpr::Var(_, x) =>
                if let Some(idx) = ctx.lookup(x) {
                    Expr::Var(idx)
                } else if let Some(&glob) = self.builtins.get(&x) {
                    Expr::Glob(glob)
                } else {
                    panic!("couldn't find a variable??")
                },
            HExpr::Val(_, HValue::Unit) =>
                Expr::Val(Value::Unit),
            HExpr::Val(_, HValue::Sample(x)) =>
                Expr::Val(Value::Sample(x)),
            HExpr::Val(_, HValue::Index(i)) =>
                Expr::Val(Value::Index(i)),
            HExpr::Val(_, _) =>
                panic!("weird value??"),
            HExpr::Annotate(_, next, _) =>
                self.translate(ctx, next),
            HExpr::Lam(_, x, next) => {
                let new_ctx = Rc::new(Ctx::Var(x, ctx));
                Expr::Lam(None, 1, self.alloc(self.translate(new_ctx, next)))
            },
            HExpr::App(_, e1, e2) => {
                let e1t = self.translate(ctx.clone(), e1);
                let e2t = self.translate(ctx, e2);
                Expr::App(self.alloc(e1t), self.alloc_slice([self.alloc(e2t)]))
            },
            HExpr::Adv(_, e) => {
                let et = self.translate(ctx, e);
                Expr::Adv(self.alloc(et))
            },
            HExpr::Lob(_, _, x, e) => {
                let new_ctx = Rc::new(Ctx::Var(x, ctx));
                let et = self.translate(new_ctx, e);
                Expr::Lob(None, self.alloc(et))
            },
            HExpr::Gen(_, e1, e2) => {
                let e1t = self.translate(ctx.clone(), e1);
                let e2t = self.translate(ctx, e2);
                Expr::Con(Con::Stream, self.alloc_slice([self.alloc(e1t), self.alloc(e2t)]))
            },
            HExpr::LetIn(_, x, _, e1, e2) => {
                let e1t = self.translate(ctx.clone(), e1);
                let new_ctx = Rc::new(Ctx::Var(x, ctx));
                let e2t = self.translate(new_ctx, e2);
                Expr::LetIn(self.alloc(e1t), self.alloc(e2t))
            },
            HExpr::Pair(_, e1, e2) => {
                let e1t = self.translate(ctx.clone(), e1);
                let e2t = self.translate(ctx, e2);
                Expr::Con(Con::Pair, self.alloc_slice([self.alloc(e1t), self.alloc(e2t)]))
            },
            HExpr::UnPair(_, x1, x2, e1, e2) => {
                // TODO: why do i have unpair instead of fst and snd
                // in the high level language? i guess somewhat
                // because i don't have patterns
                let e1t = self.translate(ctx.clone(), e1);
                let new_ctx = Rc::new(Ctx::Var(x2, Rc::new(Ctx::Var(x1, Rc::new(Ctx::Silent(ctx))))));
                let e2t = self.translate(new_ctx, e2);
                Expr::LetIn(self.alloc(e1t), self.alloc(
                    Expr::LetIn(self.alloc(
                        Expr::Op(Op::Fst, self.alloc_slice([self.alloc(
                            Expr::Var(DebruijnIndex::HERE)
                        )]))
                    ), self.alloc(
                        Expr::LetIn(self.alloc(
                            Expr::Op(Op::Snd, self.alloc_slice([self.alloc(
                                Expr::Var(DebruijnIndex::HERE.shifted())
                            )]))
                        ), self.alloc(e2t))))))
            },
            HExpr::InL(_, e) => {
                let et = self.translate(ctx, e);
                Expr::Con(Con::InL, self.alloc_slice([self.alloc(et)]))
            },
            HExpr::InR(_, e) => {
                let et = self.translate(ctx, e);
                Expr::Con(Con::InR, self.alloc_slice([self.alloc(et)]))
            },
            HExpr::Case(_, e0, x1, e1, x2, e2) => {
                let e0t = self.translate(ctx.clone(), e0);
                let new_ctx1 = Rc::new(Ctx::Var(x1, ctx.clone()));
                let e1t = self.translate(new_ctx1, e1);
                let new_ctx2 = Rc::new(Ctx::Var(x2, ctx));
                let e2t = self.translate(new_ctx2, e2);
                Expr::Case(self.alloc(e0t), self.alloc(e1t), self.alloc(e2t))
            },
            HExpr::Array(_, ref es) => {
                let est = es.iter().map(|&e| self.alloc(self.translate(ctx.clone(), e)));
                Expr::Con(Con::Array, self.alloc_slice(est))
            },
            HExpr::UnGen(_, e) => {
                let et = self.translate(ctx, e);
                Expr::Op(Op::UnGen, self.alloc_slice([self.alloc(et)]))
            },
            HExpr::Delay(_, e) => {
                // TODO: we may eventually want to do things
                // differently here depending on the clock, if we use
                // that for eg garbage collection information
                let et = self.translate(ctx, e);
                Expr::Delay(None, self.alloc(et))
            },
            HExpr::Box(_, e) => {
                let et = self.translate(ctx, e);
                Expr::Box(None, self.alloc(et))
            },
            HExpr::Unbox(_, e) => {
                let et = self.translate(ctx, e);
                Expr::Unbox(self.alloc(et))
            },
            HExpr::ClockApp(_, e, _) =>
                // for now...
                self.translate(ctx, e),
            HExpr::TypeApp(_, e, _) =>
                self.translate(ctx, e),
        }
    }

}

#[derive(Debug, Clone, Copy)]
enum RewriteType {
    WithinDelay,
    WithinLambda,
}

struct Abstracted<'a, T: ?Sized> {
    binding: &'a Expr<'a>,
    abstracted: &'a T,
}

impl RewriteType {
    fn adv_subterm_matches<'a>(&self, subterm: &Expr<'a>) -> bool {
        use RewriteType::*;
        match (self, subterm) {
            (&WithinDelay, &Expr::Var(_)) => false,
            (&WithinDelay, _) => true,
            (&WithinLambda, _) => true,
        }
    }

    fn abstract_term<'a>(&self, subterm: &'a Expr<'a>, hole_depth: u32, arena: &'a ExprArena<'a>) -> Abstracted<'a, Expr<'a>> {
        use RewriteType::*;
        match *self {
            WithinDelay => Abstracted {
                binding: arena.alloc(subterm.shifted_by_signed(-(hole_depth as i32), 0, arena)),
                abstracted: arena.alloc(Expr::Adv(arena.alloc(Expr::Var(DebruijnIndex(hole_depth)))))
            },
            WithinLambda => Abstracted {
                // hole_depth **includes** the lambda we are under
                binding: arena.alloc(Expr::Adv(arena.alloc(subterm.shifted_by_signed(-(hole_depth as i32), 0, arena)))),
                abstracted: arena.alloc(Expr::Var(DebruijnIndex(hole_depth)))
            },
        }
    }
}

impl<'a, T: ?Sized> Abstracted<'a, T> {
    /// keep binding the same, but transform abstracted
    fn map<U: ?Sized>(self, f: impl FnOnce(&'a T) -> &'a U) -> Abstracted<'a, U> {
        Abstracted {
            binding: self.binding,
            abstracted: f(self.abstracted),
        }
    }
}

impl<'a> Translator<'a> {
    // TODO: can probably make this rewriting close to a single pass, but do it inefficiently for now. might be good enough, as long as we make sure to separate out global definitions first
    //
    /// Returns the new let-binding to make (if there is one), and the corresponding abstracted expression.
    fn abstract_adv(&self, rew: RewriteType, depth: u32, expr: &'a Expr<'a>) -> Option<Abstracted<'a, Expr<'a>>> {
        use Expr::*;
        match *expr {
            Var(_) => None,
            Val(_) => None,
            Glob(_) => None,
            Lam(_, _, _) =>
                None,
            App(e0, es) =>
                if let Some(abs) = self.abstract_adv(rew, depth, e0) {
                    Some(abs.map(|ea| self.alloc(App(ea, self.slice_shifted(1, 0, es)))))
                } else if let Some(abs) = self.slice_abstracted(rew, depth, es) {
                    Some(abs.map(|args| self.alloc(App(self.alloc(e0.shifted_by(1, 0, self.arena)), args))))
                } else {
                    None
                },
            Unbox(e) =>
                Some(self.abstract_adv(rew, depth, e)?
                         .map(|ea| self.alloc(Unbox(ea)))),
            Box(_, _) =>
                None,
            Lob(_, _) =>
                None,
            LetIn(e1, e2) =>
                if let Some(abs) = self.abstract_adv(rew, depth, e1) {
                    Some(abs.map(|e1p| self.alloc(LetIn(e1p, self.alloc(e2.shifted_by(1, 0, self.arena))))))
                } else if let Some(abs) = self.abstract_adv(rew, depth + 1, e2) {
                    Some(abs.map(|e2p| self.alloc(LetIn(self.alloc(e1.shifted_by(1, 0, self.arena)), e2p))))
                } else {
                    None
                },
            Case(e0, e1, e2) =>
                if let Some(abs) = self.abstract_adv(rew, depth, e0) {
                    Some(abs.map(|e0p| self.alloc(Case(
                        e0p,
                        self.alloc(e1.shifted_by(1, 0, self.arena)),
                        self.alloc(e2.shifted_by(1, 0, self.arena))
                    ))))
                } else if let Some(abs) = self.abstract_adv(rew, depth + 1, e1) {
                    Some(abs.map(|e1p| self.alloc(Case(
                        self.alloc(e0.shifted_by(1, 0, self.arena)),
                        e1p,
                        self.alloc(e2.shifted_by(1, 0, self.arena))
                    ))))
                } else if let Some(abs) = self.abstract_adv(rew, depth + 1, e2) {
                    Some(abs.map(|e2p| self.alloc(Case(
                        self.alloc(e0.shifted_by(1, 0, self.arena)),
                        self.alloc(e1.shifted_by(1, 0, self.arena)),
                        e2p
                    ))))
                } else {
                    None
                },
            Con(con, es) =>
                if let Some(abs) = self.slice_abstracted(rew, depth, es) {
                    Some(abs.map(|args| self.alloc(Con(con, args))))
                } else {
                    None
                },
            Op(con, es) =>
                if let Some(abs) = self.slice_abstracted(rew, depth, es) {
                    Some(abs.map(|args| self.alloc(Op(con, args))))
                } else {
                    None
                },
            Delay(_, _) =>
                None,
            // aha! finally we have found you!
            Adv(e) =>
                if rew.adv_subterm_matches(e) {
                    Some(rew.abstract_term(e, depth, self.arena))
                } else {
                    None
                },
        }
    }

    fn slice_shifted(&self, by: u32, depth: u32, exprs: &[&'a Expr<'a>]) -> &'a [&'a Expr<'a>] {
        self.arena.alloc_slice_r(exprs.iter().map(|e| self.alloc(e.shifted_by(by, depth, self.arena))))
    }

    fn slice_abstracted(&self, rew: RewriteType, depth: u32, exprs: &[&'a Expr<'a>]) -> Option<Abstracted<'a, [&'a Expr<'a>]>> {
        for (i, arg) in exprs.iter().enumerate() {
            if let Some(abs) = self.abstract_adv(rew, depth, arg) {
                // this is a doozy
                let args_abstracted = self.arena.alloc_slice_r(exprs.iter().enumerate().map(|(j, argj)|
                    if i == j {
                        abs.abstracted
                    } else {
                        self.alloc(argj.shifted_by(1, 0, self.arena))
                    }
                ));
                return Some(Abstracted { binding: abs.binding, abstracted: args_abstracted });
            }
        }
        None
    }

    fn build_lets(&self, bindings: &[&'a Expr<'a>], body: &'a Expr<'a>) -> &'a Expr<'a> {
        bindings.iter().rev().fold(body, |so_far, to_bind| {
            self.alloc(Expr::LetIn(to_bind, so_far))
        })
    }

    // TODO: this doesn't actually work for us. in particular,
    // "delay(K[adv t]) --> let x = t in delay(K[adv x])" means the
    // stuff in each adv only gets bumped up through one delay, which
    // may not be enough in our setting!
    //
    // TODO: optimize allocation for when nothing is rewritten?
    // probably doing a bunch we don't need to be...
    pub fn rewrite(&self, expr: &'a Expr<'a>) -> &'a Expr<'a> {
        use Expr::*;
        match *expr {
            Var(_) => expr,
            Val(_) => expr,
            Glob(_) => expr,
            Lam(ref vars, arity, e) => {
                let mut ep = self.rewrite(e);
                let mut bindings = Vec::new();
                while let Some(abs) = self.abstract_adv(RewriteType::WithinLambda, arity, ep) {
                    bindings.push(abs.binding);
                    ep = abs.abstracted;
                }
                self.build_lets(&bindings, self.alloc(Lam(vars.clone(), arity, ep)))
            },
            App(e, es) => self.alloc(App(self.rewrite(e), self.slice_rewrite(es))),
            Unbox(e) => self.alloc(Unbox(self.rewrite(e))),
            Box(ref vars, e) => self.alloc(Box(vars.clone(), self.rewrite(e))),
            Lob(ref vars, e) => self.alloc(Lob(vars.clone(), self.rewrite(e))),
            LetIn(e1, e2) => self.alloc(LetIn(self.rewrite(e1), self.rewrite(e2))),
            Case(e0, e1, e2) => self.alloc(Case(self.rewrite(e0), self.rewrite(e1), self.rewrite(e2))),
            Con(con, es) => self.alloc(Con(con, self.slice_rewrite(es))),
            Op(op, es) => self.alloc(Op(op, self.slice_rewrite(es))),
            Delay(ref vars, e) => {
                let mut ep = self.rewrite(e);
                let mut bindings = Vec::new();
                while let Some(abs) = self.abstract_adv(RewriteType::WithinDelay, 0, ep) {
                    bindings.push(abs.binding);
                    ep = abs.abstracted;
                }
                self.build_lets(&bindings, self.alloc(Delay(vars.clone(), ep)))
            },
            Adv(e) => self.alloc(Adv(self.rewrite(e))),
        }
    }

    fn slice_rewrite(&self, exprs: &[&'a Expr<'a>]) -> &'a [&'a Expr<'a>] {
        self.arena.alloc_slice_r(exprs.iter().map(|e| self.rewrite(e)))
    }
}

#[derive(Debug, Clone)]
pub enum VarSetThunk {
    Empty,
    Var(DebruijnIndex),
    Union(Rc<VarSetThunk>, Rc<VarSetThunk>),
    Shift(u32, Rc<VarSetThunk>),
}

impl VarSetThunk {
    fn to_var_set(&self) -> VarSet {
        use VarSetThunk::*;
        match *self {
            Empty => HashSet::new(),
            Var(i) => { let mut vs = HashSet::new(); vs.insert(i); vs },
            Union(ref t1, ref t2) => {
                let mut vs = t1.to_var_set();
                vs.extend(t2.to_var_set().into_iter());
                vs
            },
            Shift(n, ref t) =>
                t.to_var_set().into_iter().filter_map(|i| if i.is_within(n) { None } else { Some(i.shifted_by_signed(-(n as i32))) }).collect(),
        }
    }
}

impl<'a> Translator<'a> {

    pub fn annotate_used_vars(&self, expr: &'a Expr<'a>) -> (&'a Expr<'a>, VarSetThunk) {
        use Expr::*;
        match *expr {
            Var(i) =>
                (expr, VarSetThunk::Var(i)),
            Val(_) =>
                (expr, VarSetThunk::Empty),
            Glob(_) =>
                (expr, VarSetThunk::Empty),
            Lam(None, arity, e) => {
                let (ep, vs) = self.annotate_used_vars(e);
                let shifted = VarSetThunk::Shift(arity, Rc::new(vs));
                (self.alloc(Lam(Some(shifted.to_var_set()), arity, ep)), shifted)
            },
            Lam(Some(_), _, _) =>
                panic!("why are you re-annotating???"),
            App(e1, es) => {
                let (e1p, vs1) = self.annotate_used_vars(e1);
                let mut vss = Rc::new(vs1);
                let esp = self.arena.alloc_slice_r(es.iter().map(|e| {
                    let (ep, vs) = self.annotate_used_vars(e);
                    vss = VarSetThunk::Union(vss.clone(), vs.into()).into();
                    ep
                }));
                (self.alloc(App(e1p, esp)), (*vss).clone())
            },
            Unbox(e) => {
                let (ep, vs) = self.annotate_used_vars(e);
                (self.alloc(Unbox(ep)), vs)
            },
            Box(None, e) => {
                let (ep, vs) = self.annotate_used_vars(e);
                (self.alloc(Box(Some(vs.to_var_set()), ep)), vs)
            },
            Box(Some(_), _) =>
                panic!("why are you re-annotating???"),
            Lob(None, e) => {
                let (ep, vs) = self.annotate_used_vars(e);
                let shifted = VarSetThunk::Shift(1, Rc::new(vs));
                (self.alloc(Lob(Some(shifted.to_var_set()), ep)), shifted)
            },
            Lob(Some(_), _) =>
                panic!("why are you re-annotating???"),
            LetIn(e1, e2) => {
                let (e1p, vs1) = self.annotate_used_vars(e1);
                let (e2p, vs2) = self.annotate_used_vars(e2);
                (self.alloc(LetIn(e1p, e2p)), VarSetThunk::Union(vs1.into(), VarSetThunk::Shift(1, vs2.into()).into()))
            },
            Case(e0, e1, e2) => {
                let (e0p, vs0) = self.annotate_used_vars(e0);
                let (e1p, vs1) = self.annotate_used_vars(e1);
                let (e2p, vs2) = self.annotate_used_vars(e2);
                (self.alloc(Case(e0p, e1p, e2p)), VarSetThunk::Union(vs0.into(), VarSetThunk::Shift(1, VarSetThunk::Union(vs1.into(), vs2.into()).into()).into()))
            },
            Con(con, es) => {
                let mut vss = Rc::new(VarSetThunk::Empty);
                let esp = self.arena.alloc_slice_r(es.iter().map(|e| {
                    let (ep, vs) = self.annotate_used_vars(e);
                    vss = VarSetThunk::Union(vss.clone(), vs.into()).into();
                    ep
                }));
                (self.alloc(Con(con, esp)), (*vss).clone())
            },
            Op(op, es) => {
                let mut vss = Rc::new(VarSetThunk::Empty);
                let esp = self.arena.alloc_slice_r(es.iter().map(|e| {
                    let (ep, vs) = self.annotate_used_vars(e);
                    vss = VarSetThunk::Union(vss.clone(), vs.into()).into();
                    ep
                }));
                (self.alloc(Op(op, esp)), (*vss).clone())
            },
            Adv(e) => {
                let (ep, vs) = self.annotate_used_vars(e);
                (self.alloc(Adv(ep)), vs)
            },
            Delay(None, e) => {
                let (ep, vs) = self.annotate_used_vars(e);
                (self.alloc(Delay(Some(vs.to_var_set()), ep)), vs)
            },
            Delay(Some(_), _) =>
                panic!("why are you re-annotating???"),
        }
    }
}


