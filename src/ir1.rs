use std::{rc::Rc, collections::{HashMap, HashSet}, fmt};

use imbl as im;
use num::rational::Ratio;

use crate::expr::{Expr as HExpr, Symbol, Value as HValue, Binop as HBinop};
use crate::util::ArenaPlus;

// okay, for real, what am i really getting here over just using u32...
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DebruijnIndex(pub u32);

impl fmt::Debug for DebruijnIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl DebruijnIndex {
    pub const HERE: DebruijnIndex = DebruijnIndex(0);

    pub const fn shifted(self) -> DebruijnIndex {
        DebruijnIndex(self.0 + 1)
    }

    pub const fn shifted_by(self, by: u32) -> DebruijnIndex {
        DebruijnIndex(self.0 + by)
    }

    pub fn shifted_by_signed(self, by: i32) -> DebruijnIndex {
        if let Some(new) = self.0.checked_add_signed(by) {
            DebruijnIndex(new)
        } else {
            panic!("debruijn underflow")
        }
    }

    pub const fn is_within(&self, depth: u32) -> bool {
        self.0 < depth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Global(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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
    ClockEx,
}

impl Con {
    // returns None if variable arity
    #[allow(unused)]
    fn arity(&self) -> Option<u8> {
        match *self {
            Con::Stream => Some(2),
            Con::Array => None,
            Con::Pair => Some(2),
            Con::InL => Some(1),
            Con::InR => Some(1),
            Con::ClockEx => Some(2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Op {
    Const(Value),
    FAdd,
    FSub,
    FMul,
    FDiv,
    FGt,
    FGe,
    FLt,
    FLe,
    FEq,
    FNe,
    Sin,
    Cos,
    Pi,
    IAdd,
    ISub,
    IMul,
    IDiv,
    Shl,
    Shr,
    And,
    Xor,
    Or,
    IGt,
    IGe,
    ILt,
    ILe,
    IEq,
    INe,
    ReinterpF2I,
    ReinterpI2F,
    CastI2F,
    // TODO: make this a more informative index?
    Proj(u32),
    UnGen,
    // TODO: coalesce (some of?) these
    AllocAndFill,
    AllocF32,
    AllocI32,
    BuildClosure(Global),
    LoadGlobal(Global),
    DerefF32,
    DerefI32,
    ApplyCoeff(Ratio<u32>),
    SinceLastTickStream,
    Advance,
    Wait,
    Schedule,
    MakeClock(f32),
    GetClock(u32),
}

impl Op {
    fn from_binop(op: HBinop) -> Op {
        match op {
            HBinop::FMul => Op::FMul,
            HBinop::FDiv => Op::FDiv,
            HBinop::FAdd => Op::FAdd,
            HBinop::FSub => Op::FSub,
            HBinop::FGt => Op::FGt,
            HBinop::FGe => Op::FGe,
            HBinop::FLt => Op::FLt,
            HBinop::FLe => Op::FLe,
            HBinop::FEq => Op::FEq,
            HBinop::FNe => Op::FNe,
            HBinop::Shl => Op::Shl,
            HBinop::Shr => Op::Shr,
            HBinop::And => Op::And,
            HBinop::Xor => Op::Xor,
            HBinop::Or => Op::Or,
            HBinop::IMul => Op::IMul,
            HBinop::IDiv => Op::IDiv,
            HBinop::IAdd => Op::IAdd,
            HBinop::ISub => Op::ISub,
            HBinop::IGt => Op::IGt,
            HBinop::IGe => Op::IGe,
            HBinop::ILt => Op::ILt,
            HBinop::ILe => Op::ILe,
            HBinop::IEq => Op::IEq,
            HBinop::INe => Op::INe,
        }
    }
}

impl Op {
    #[allow(unused)]
    fn arity(&self) -> Option<u8> {
        match *self {
            Op::Const(_) => Some(0),
            Op::FAdd => Some(2),
            Op::FSub => Some(2),
            Op::FMul => Some(2),
            Op::FDiv => Some(2),
            Op::FGt => Some(2),
            Op::FGe => Some(2),
            Op::FLt => Some(2),
            Op::FLe => Some(2),
            Op::FEq => Some(2),
            Op::FNe => Some(2),
            Op::Sin => Some(1),
            Op::Cos => Some(1),
            Op::Pi => Some(0),
            Op::IAdd => Some(2),
            Op::ISub => Some(2),
            Op::IMul => Some(2),
            Op::IDiv => Some(2),
            Op::Shl => Some(2),
            Op::Shr => Some(2),
            Op::And => Some(2),
            Op::Xor => Some(2),
            Op::Or => Some(2),
            Op::IGt => Some(2),
            Op::IGe => Some(2),
            Op::ILt => Some(2),
            Op::ILe => Some(2),
            Op::IEq => Some(2),
            Op::INe => Some(2),
            Op::ReinterpF2I => Some(1),
            Op::ReinterpI2F => Some(1),
            Op::CastI2F => Some(1),
            Op::Proj(_) => Some(1),
            Op::UnGen => Some(1),
            Op::AllocAndFill => None,
            Op::AllocI32 => Some(1),
            Op::AllocF32 => Some(1),
            Op::BuildClosure(_) => None,
            Op::DerefI32 => Some(1),
            Op::DerefF32 => Some(1),
            Op::LoadGlobal(_) => Some(0),
            Op::ApplyCoeff(_) => Some(1),
            Op::SinceLastTickStream => Some(1),
            Op::Advance => Some(1),
            Op::Wait => Some(1),
            Op::Schedule => Some(3),
            Op::MakeClock(_) => Some(0),
            Op::GetClock(_) => Some(0),
        }
    }
}

// TODO: change this out for some sort of bit set -- particularly one
// that has an optimization for small sizes, bc i imagine that will be
// the *very* common case
type VarsUsed = Vec<DebruijnIndex>;

#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    Var(DebruijnIndex),
    Val(Value),
    Glob(Global),
    Lam(Option<VarsUsed>, u32, &'a Expr<'a>),
    App(&'a Expr<'a>, &'a [&'a Expr<'a>]),
    Unbox(&'a Expr<'a>),
    Box(Option<VarsUsed>, &'a Expr<'a>),
    Lob(Option<VarsUsed>, &'a Expr<'a>),
    LetIn(&'a Expr<'a>, &'a Expr<'a>),
    If(&'a Expr<'a>, &'a Expr<'a>, &'a Expr<'a>),
    Con(Con, &'a [&'a Expr<'a>]),
    Op(Op, &'a [&'a Expr<'a>]),
    // These two are initially treated separately from Force and
    // Suspend because after the initial translation, we have to do
    // the rewriting step of pulling Advances out of Delays and
    // Lams... and maybe we will want to handle allocation/GC
    // differently for them? idk
    Delay(Option<VarsUsed>, &'a Expr<'a>),
    Adv(&'a Expr<'a>),
}

impl<'a> Expr<'a> {
    // i don't use these? really? i guess i'll probably use them when
    // writing optimizations
    #[allow(unused)]
    fn shifted_by<'b>(&self, by: u32, depth: u32, arena: &'b ArenaPlus<'b, Expr<'b>>) -> Expr<'b> {
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
            If(e0, e1, e2) => If(
                arena.alloc(e0.shifted_by(by, depth, arena)),
                arena.alloc(e1.shifted_by(by, depth, arena)),
                arena.alloc(e2.shifted_by(by, depth, arena))
            ),
            Con(con, es) => Con(con, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by(by, depth, arena))))),
            Op(op, es) => Op(op, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by(by, depth, arena))))),
            Delay(ref vs, e) => Delay(vs.clone(), arena.alloc(e.shifted_by(by, depth, arena))),
            Adv(e) => Adv(arena.alloc(e.shifted_by(by, depth, arena))),
        }
    }

    #[allow(unused)]
    fn shifted_by_signed<'b>(&self, by: i32, depth: u32, arena: &'b ArenaPlus<'b, Expr<'b>>) -> Expr<'b> {
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
            If(e0, e1, e2) => If(
                arena.alloc(e0.shifted_by_signed(by, depth, arena)),
                arena.alloc(e1.shifted_by_signed(by, depth, arena)),
                arena.alloc(e2.shifted_by_signed(by, depth, arena))
            ),
            Con(con, es) => Con(con, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by_signed(by, depth, arena))))),
            Op(op, es) => Op(op, arena.alloc_slice_r(es.iter().map(|e| arena.alloc(e.shifted_by_signed(by, depth, arena))))),
            Delay(ref vs, e) => Delay(vs.clone(), arena.alloc(e.shifted_by_signed(by, depth, arena))),
            Adv(e) => Adv(arena.alloc(e.shifted_by_signed(by, depth, arena))),
        }
    }
}
            


pub struct Translator<'a> {
    pub globals: HashMap<Symbol, Global>,
    pub global_clocks: HashMap<Symbol, Global>,
    pub arena: &'a ArenaPlus<'a, Expr<'a>>,
}

pub enum Ctx {
    Empty,
    TermVar(Symbol, Rc<Ctx>),
    ClockVar(Symbol, Rc<Ctx>),
    Silent(Rc<Ctx>),
}

impl Ctx {
    fn lookup_termvar(&self, x: Symbol) -> Option<DebruijnIndex> {
        match *self {
            Ctx::Empty =>
                None,
            Ctx::TermVar(y, _) if x == y =>
                Some(DebruijnIndex::HERE),
            Ctx::TermVar(_, ref next) =>
                next.lookup_termvar(x).map(|i| i.shifted()),
            Ctx::ClockVar(_, ref next) =>
                next.lookup_termvar(x).map(|i| i.shifted()),
            Ctx::Silent(ref next) =>
                next.lookup_termvar(x).map(|i| i.shifted()),
        }
    }

    fn lookup_clockvar(&self, x: Symbol) -> Option<DebruijnIndex> {
        match *self {
            Ctx::Empty =>
                None,
            Ctx::TermVar(_, ref next) =>
                next.lookup_clockvar(x).map(|i| i.shifted()),
            Ctx::ClockVar(y, _) if x == y =>
                Some(DebruijnIndex::HERE),
            Ctx::ClockVar(_, ref next) =>
                next.lookup_clockvar(x).map(|i| i.shifted()),
            Ctx::Silent(ref next) =>
                next.lookup_clockvar(x).map(|i| i.shifted()),
        }
    }
}

fn binop_types(b: HBinop) -> (Op, Op, Op) {
    match b {
        HBinop::FMul |
        HBinop::FDiv |
        HBinop::FAdd |
        HBinop::FSub =>
            (Op::DerefF32, Op::DerefF32, Op::AllocF32),
        HBinop::FGt |
        HBinop::FGe |
        HBinop::FLt |
        HBinop::FLe |
        HBinop::FEq |
        HBinop::FNe =>
            (Op::DerefF32, Op::DerefF32, Op::AllocI32),
        HBinop::Shl |
        HBinop::Shr |
        HBinop::And |
        HBinop::Xor |
        HBinop::Or |
        HBinop::IMul |
        HBinop::IDiv |
        HBinop::IAdd |
        HBinop::ISub |
        HBinop::IGt |
        HBinop::IGe |
        HBinop::ILt |
        HBinop::ILe |
        HBinop::IEq |
        HBinop::INe =>
            (Op::DerefI32, Op::DerefI32, Op::AllocI32),
    }
}

impl<'a> Translator<'a> {
    fn alloc(&self, e: Expr<'a>) -> &'a Expr<'a> {
        self.arena.alloc(e)
    }

    fn alloc_slice(&self, it: impl IntoIterator<Item=&'a Expr<'a>>) -> &'a [&'a Expr<'a>] {
        self.arena.alloc_slice(it)
    }

    fn make_alloc_f32(&self, e: &'a Expr<'a>) -> Expr<'a> {
        Expr::Op(Op::AllocF32, self.alloc_slice([e]))
    }

    fn make_alloc_i32(&self, e: &'a Expr<'a>) -> Expr<'a> {
        Expr::Op(Op::AllocI32, self.alloc_slice([e]))
    }

    pub fn translate<'b, R>(&self, ctx: Rc<Ctx>, expr: &'b HExpr<'b, R>) -> Expr<'a> {
        match *expr {
            HExpr::Var(_, x) =>
                if let Some(idx) = ctx.lookup_termvar(x) {
                    Expr::Var(idx)
                } else if let Some(&glob) = self.globals.get(&x) {
                    Expr::Glob(glob)
                } else {
                    panic!("couldn't find a variable??")
                },
            HExpr::Val(_, HValue::Unit) =>
                self.make_alloc_i32(self.alloc(Expr::Val(Value::Unit))),
            HExpr::Val(_, HValue::Sample(x)) =>
                self.make_alloc_f32(self.alloc(Expr::Val(Value::Sample(x)))),
            HExpr::Val(_, HValue::Index(i)) =>
                self.make_alloc_i32(self.alloc(Expr::Val(Value::Index(i)))),
            HExpr::Annotate(_, next, _) =>
                self.translate(ctx, next),
            HExpr::Lam(_, x, next) => {
                let new_ctx = Rc::new(Ctx::TermVar(x, ctx));
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
                let new_ctx = Rc::new(Ctx::TermVar(x, ctx));
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
                let new_ctx = Rc::new(Ctx::TermVar(x, ctx));
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
                let new_ctx = Rc::new(Ctx::TermVar(x2, Rc::new(Ctx::TermVar(x1, Rc::new(Ctx::Silent(ctx))))));
                let e2t = self.translate(new_ctx, e2);
                Expr::LetIn(self.alloc(e1t), self.alloc(
                    Expr::LetIn(self.alloc(
                        Expr::Op(Op::Proj(0), self.alloc_slice([self.alloc(
                            Expr::Var(DebruijnIndex::HERE)
                        )]))
                    ), self.alloc(
                        Expr::LetIn(self.alloc(
                            Expr::Op(Op::Proj(1), self.alloc_slice([self.alloc(
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
                let new_ctx = Rc::new(Ctx::Silent(ctx));
                let new_ctx1 = Rc::new(Ctx::TermVar(x1, new_ctx.clone()));
                let e1t = self.translate(new_ctx1, e1);
                let new_ctx2 = Rc::new(Ctx::TermVar(x2, new_ctx));
                let e2t = self.translate(new_ctx2, e2);
                // TODO: should really make this the same abstraction level as InL/InR...
                Expr::LetIn(
                    self.alloc(e0t),
                    self.alloc(Expr::LetIn(
                        self.alloc(Expr::Op(Op::Proj(1), self.alloc_slice([self.alloc(Expr::Var(DebruijnIndex::HERE))]))),
                        self.alloc(Expr::If(
                            self.alloc(Expr::Op(Op::Proj(0), self.alloc_slice([self.alloc(Expr::Var(DebruijnIndex::HERE.shifted()))]))),
                            self.alloc(e1t),
                            self.alloc(e2t)
                        ))
                    ))
                )
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
            HExpr::ClockApp(_, e, c) => {
                // TODO: factor out this lookup logic
                let c_var_expr = self.alloc(if let Some(idx) = ctx.lookup_clockvar(c.var) {
                    Expr::Var(idx)
                } else if let Some(&glob) = self.global_clocks.get(&c.var) {
                    Expr::Glob(glob)
                } else {
                    panic!("couldn't find clock var??")
                });
                let c_expr = self.alloc(Expr::Op(Op::ApplyCoeff(c.coeff), self.alloc_slice([c_var_expr])));
                let et = self.alloc(self.translate(ctx, e));
                Expr::App(et, self.alloc_slice([c_expr]))
            },
            HExpr::TypeApp(_, e, _) =>
                self.translate(ctx, e),
            HExpr::Binop(_, op, e1, e2) => {
                let (a1o, a2o, ro) = binop_types(op);
                let e1p = Expr::Op(a1o, self.alloc_slice([self.alloc(self.translate(ctx.clone(), e1))]));
                let e2p = Expr::Op(a2o, self.alloc_slice([self.alloc(self.translate(ctx, e2))]));
                Expr::Op(ro, self.alloc_slice([self.alloc(
                    Expr::Op(Op::from_binop(op), self.alloc_slice([self.alloc(e1p), self.alloc(e2p)]))
                )]))
            },
            HExpr::ExIntro(_, c, e) => {
                // TODO: factor out this lookup logic
                let c_var_expr = self.alloc(if let Some(idx) = ctx.lookup_clockvar(c.var) {
                    Expr::Var(idx)
                } else if let Some(&glob) = self.global_clocks.get(&c.var) {
                    Expr::Glob(glob)
                } else {
                    panic!("couldn't find clock var??")
                });
                let c_expr = self.alloc(Expr::Op(Op::ApplyCoeff(c.coeff), self.alloc_slice([c_var_expr])));
                let et = self.alloc(self.translate(ctx, e));
                Expr::Con(Con::ClockEx, self.alloc_slice([c_expr, et]))
            },
            HExpr::ExElim(_, c, x, e1, e2) => {
                let e1t = self.alloc(self.translate(ctx.clone(), e1));
                let new_ctx = Ctx::TermVar(x, Ctx::ClockVar(c, Ctx::Silent(ctx).into()).into()).into();
                let e2t = self.alloc(self.translate(new_ctx, e2));
                let piece = |i: u32| {
                    self.alloc(Expr::Op(Op::Proj(i), self.alloc_slice([
                        self.alloc(Expr::Var(DebruijnIndex(i)))
                    ])))
                };
                Expr::LetIn(e1t, self.alloc(
                    Expr::LetIn(piece(0), self.alloc(
                        Expr::LetIn(piece(1), e2t)))))
            },
            HExpr::ClockLam(_, x, e) => {
                let new_ctx = Rc::new(Ctx::ClockVar(x, ctx));
                let et = self.translate(new_ctx, e);
                Expr::Lam(None, 1, self.alloc(et))
            },
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

    fn abstract_term<'a>(&self, subterm: &'a Expr<'a>, hole_depth: u32, arena: &'a ArenaPlus<'a, Expr<'a>>) -> Abstracted<'a, Expr<'a>> {
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
            If(e0, e1, e2) =>
                if let Some(abs) = self.abstract_adv(rew, depth, e0) {
                    Some(abs.map(|e0p| self.alloc(If(
                        e0p,
                        self.alloc(e1.shifted_by(1, 0, self.arena)),
                        self.alloc(e2.shifted_by(1, 0, self.arena))
                    ))))
                } else if let Some(abs) = self.abstract_adv(rew, depth, e1) {
                    Some(abs.map(|e1p| self.alloc(If(
                        self.alloc(e0.shifted_by(1, 0, self.arena)),
                        e1p,
                        self.alloc(e2.shifted_by(1, 0, self.arena))
                    ))))
                } else if let Some(abs) = self.abstract_adv(rew, depth, e2) {
                    Some(abs.map(|e2p| self.alloc(If(
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
    #[allow(unused)]
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
            If(e0, e1, e2) => self.alloc(If(self.rewrite(e0), self.rewrite(e1), self.rewrite(e2))),
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
    fn to_var_set(&self) -> HashSet<DebruijnIndex> {
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

    fn to_vars_used(&self) -> VarsUsed {
        self.to_var_set().into_iter().collect()
    }
}

type EnvMap = im::HashMap<DebruijnIndex, DebruijnIndex>;

fn mk_env(used: &[DebruijnIndex]) -> EnvMap {
    used.iter().copied().zip((0..).map(DebruijnIndex)).collect()
}

fn remap_var(env_map: &EnvMap, locals: u32, i: DebruijnIndex) -> DebruijnIndex {
    if i.is_within(locals) {
        i
    } else {
        env_map.get(&i.shifted_by_signed(-(locals as i32)))
               .expect("couldn't find variable in annotated environment?")
               .shifted_by(locals)
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
                (self.alloc(Lam(Some(shifted.to_vars_used()), arity, ep)), shifted)
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
                (self.alloc(Box(Some(vs.to_vars_used()), ep)), vs)
            },
            Box(Some(_), _) =>
                panic!("why are you re-annotating???"),
            Lob(None, e) => {
                let (ep, vs) = self.annotate_used_vars(e);
                let shifted = VarSetThunk::Shift(1, Rc::new(vs));
                (self.alloc(Lob(Some(shifted.to_vars_used()), ep)), shifted)
            },
            Lob(Some(_), _) =>
                panic!("why are you re-annotating???"),
            LetIn(e1, e2) => {
                let (e1p, vs1) = self.annotate_used_vars(e1);
                let (e2p, vs2) = self.annotate_used_vars(e2);
                (self.alloc(LetIn(e1p, e2p)), VarSetThunk::Union(vs1.into(), VarSetThunk::Shift(1, vs2.into()).into()))
            },
            If(e0, e1, e2) => {
                let (e0p, vs0) = self.annotate_used_vars(e0);
                let (e1p, vs1) = self.annotate_used_vars(e1);
                let (e2p, vs2) = self.annotate_used_vars(e2);
                (self.alloc(If(e0p, e1p, e2p)), VarSetThunk::Union(vs0.into(), VarSetThunk::Union(vs1.into(), vs2.into()).into()))
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
                (self.alloc(Delay(Some(vs.to_vars_used()), ep)), vs)
            },
            Delay(Some(_), _) =>
                panic!("why are you re-annotating???"),
        }
    }

    pub fn shift(&self, expr: &'a Expr<'a>, depth: u32, locals: u32, env_map: &EnvMap) -> &'a Expr<'a> {
        use Expr::*;
        match *expr {
            Var(i) =>
                self.alloc(Var(remap_var(env_map, locals, i))),
            Val(_) =>
                expr,
            Glob(_) =>
                expr,
            Lam(Some(ref used), arity, e) => {
                let new_depth = depth + locals;
                let ep = self.shift(e, new_depth, arity, &mk_env(&used));
                let used_remapped = used.iter().map(|&i| remap_var(env_map, locals, i)).collect();
                self.alloc(Lam(Some(used_remapped), arity, ep))
            },
            Lam(None, _, _) =>
                panic!("must annotate before shifting!"),
            App(e1, es) =>
                self.alloc(App(self.shift(e1, depth, locals, env_map),
                               self.arena.alloc_slice_r(es.iter().map(|e| self.shift(e, depth, locals, env_map))))),
            Unbox(e) =>
                self.alloc(Unbox(self.shift(e, depth, locals, env_map))),
            Box(Some(ref used), e) => {
                let new_depth = depth + locals;
                let ep = self.shift(e, new_depth, 0, &mk_env(&used));
                let used_remapped = used.iter().map(|&i| remap_var(env_map, locals, i)).collect();
                self.alloc(Box(Some(used_remapped), ep))
            },
            Box(None, _) =>
                panic!("must annotate before shifting!"),
            Lob(Some(ref used), e) => {
                let new_depth = depth + locals;
                let ep = self.shift(e, new_depth, 1, &mk_env(&used));
                let used_remapped = used.iter().map(|&i| remap_var(env_map, locals, i)).collect();
                self.alloc(Lob(Some(used_remapped), ep))
            },
            Lob(None, _) =>
                panic!("must annotate before shifting!"),
            LetIn(e1, e2) => {
                let e1p = self.shift(e1, depth, locals, env_map);
                let e2p = self.shift(e2, depth, locals + 1, env_map);
                self.alloc(LetIn(e1p, e2p))
            },
            If(e0, e1, e2) => {
                let e0p = self.shift(e0, depth, locals, env_map);
                let e1p = self.shift(e1, depth, locals, env_map);
                let e2p = self.shift(e2, depth, locals, env_map);
                self.alloc(If(e0p, e1p, e2p))
            },
            Con(con, es) =>
                self.alloc(Con(con, self.arena.alloc_slice_r(es.iter().map(|e| self.shift(e, depth, locals, env_map))))),
            Op(op, es) =>
                self.alloc(Op(op, self.arena.alloc_slice_r(es.iter().map(|e| self.shift(e, depth, locals, env_map))))),
            Delay(Some(ref used), e) => {
                let new_depth = depth + locals;
                let ep = self.shift(e, new_depth, 0, &mk_env(&used));
                let used_remapped = used.iter().map(|&i| remap_var(env_map, locals, i)).collect();
                self.alloc(Delay(Some(used_remapped), ep))
            },
            Delay(None, _) =>
               panic!("must annotate before shifting!"),
            Adv(e) =>
                self.alloc(Adv(self.shift(e, depth, locals, env_map))),
        }
    }
}


