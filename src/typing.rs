use core::fmt;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::rc::Rc;
use std::fmt::Write;

use num::{One, rational::Ratio};
use string_interner::DefaultStringInterner;
use indenter::{Format, indented, Indented};

use crate::expr::{Binop, Expr, SourceFile, Symbol, TopLevelDefKind, Value};
use crate::util::parenthesize;

// eventually this will get more complicated...
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ArraySize(usize);

impl ArraySize {
    pub fn from_const(n: usize) -> ArraySize {
        ArraySize(n)
    }

    pub fn as_const(&self) -> Option<usize> {
        Some(self.0)
    }
}

impl fmt::Display for ArraySize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Clock {
    pub coeff: Ratio<u32>,
    pub var: Symbol,
}

impl PartialOrd for Clock {
    fn partial_cmp(&self, other: &Clock) -> Option<Ordering> {
        if self.var == other.var {
            Some(self.coeff.cmp(&other.coeff))
        } else {
            None
        }
    }
}

impl Clock {
    pub fn pretty<'a>(&'a self, interner: &'a DefaultStringInterner) -> PrettyClock<'a> {
        PrettyClock { interner, clock: self }
    }

    fn from_var(var: Symbol) -> Clock {
        Clock { coeff: One::one(), var }
    }

    // TODO: figure out better name for this
    #[allow(unused)]
    pub fn compose(&self, other: &Clock) -> Option<Clock> {
        if self.var == other.var {
            Some(Clock { coeff: (self.coeff.recip() + other.coeff.recip()).recip(), var: self.var })
        } else {
            None
        }
    }

    // TODO: figure out better name for this
    pub fn uncompose(&self, other: &Clock) -> Option<Clock> {
        if self.var == other.var {
            Some(Clock { coeff: (self.coeff.recip() - other.coeff.recip()).recip(), var: self.var })
        } else {
            None
        }
    }

    fn substitute(&self, for_: Symbol, other: &Clock) -> Clock {
        if self.var == for_ {
            Clock { coeff: self.coeff * other.coeff, var: other.var }
        } else {
            *self
        }
    }
}

pub struct PrettyClock<'a> {
    interner: &'a DefaultStringInterner,
    clock: &'a Clock,
}

impl<'a> fmt::Display for PrettyClock<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.clock.coeff.is_one() {
            write!(f, "{} ", self.clock.coeff)?;
        }
        write!(f, "{}", self.interner.resolve(self.clock.var).unwrap())
    }
}

pub struct PrettyTiming<'a> {
    interner: &'a DefaultStringInterner,
    timing: &'a [Clock],
}

impl<'a> fmt::Display for PrettyTiming<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let num_clocks = self.timing.len();
        for (i, clock) in self.timing.into_iter().enumerate() {
            write!(f, "{}", PrettyClock { interner: self.interner, clock })?;
            if i + 1 < num_clocks {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Clock,
    Type,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Kind::Clock => write!(f, "clock"),
            Kind::Type => write!(f, "type"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Type {
    Unit,
    Sample,
    Index,
    Stream(Clock, Box<Type>),
    Function(Box<Type>, Box<Type>),
    Product(Box<Type>, Box<Type>),
    Sum(Box<Type>, Box<Type>),
    Later(Clock, Box<Type>),
    Array(Box<Type>, ArraySize),
    Box(Box<Type>),
    Forall(Symbol, Kind, Box<Type>), // forall (x : k). ty
    TypeVar(Symbol),
    Exists(Symbol, Box<Type>),
}

fn mk_fresh(prefix: Symbol, interner: &mut DefaultStringInterner) -> Symbol {
    // TODO: this is super hacky and slow
    let prefix_str = interner.resolve(prefix).unwrap();
    let mut i = 0;
    loop {
        let poss = format!("{}{}", prefix_str, i);
        if interner.get(poss.as_str()).is_none() {
            return interner.get_or_intern(poss);
        }
        i += 1;
    }
}

enum ToSubst {
    Type(Type),
    Clock(Clock),
}

impl ToSubst {
    fn from_var(var: Symbol, kind: Kind) -> ToSubst {
        match kind {
            Kind::Clock => ToSubst::Clock(Clock::from_var(var)),
            Kind::Type => ToSubst::Type(Type::TypeVar(var)),
        }
    }
}

impl Type {
    pub fn pretty<'a>(&'a self, interner: &'a DefaultStringInterner) -> PrettyType<'a> {
        PrettyType { interner, ty: self }
    }

    fn is_stable(&self) -> bool {
        match *self {
            Type::Unit => true,
            Type::Sample => true,
            Type::Index => true,
            Type::Stream(_, _) => false,
            Type::Function(_, _) => false,
            Type::Product(ref ty1, ref ty2) => ty1.is_stable() && ty2.is_stable(),
            Type::Sum(ref ty1, ref ty2) => ty1.is_stable() && ty2.is_stable(),
            Type::Later(_, _) => false,
            Type::Array(ref ty, _) => ty.is_stable(),
            Type::Box(_) => true,
            // TODO: is this right? well, it's surely not unsafe, right?
            Type::Forall(_, _, _) => false,
            // TODO: perhaps if we add constraints...
            Type::TypeVar(_) => false,
            // TODO: is this right? well, it's surely not unsafe, right?
            Type::Exists(_, _) => false,
        }
    }

    fn subst(&self, x: Symbol, ts: &ToSubst, interner: &mut DefaultStringInterner) -> Type {
        match *self {
            Type::Unit =>
                Type::Unit,
            Type::Sample =>
                Type::Sample,
            Type::Index =>
                Type::Index,
            Type::Stream(d, ref ty) => {
                let d_subst = if let &ToSubst::Clock(ref c) = ts {
                    d.substitute(x, c)
                } else { d };
                Type::Stream(d_subst, Box::new(ty.subst(x, ts, interner)))
            },
            Type::Function(ref ty1, ref ty2) =>
                Type::Function(Box::new(ty1.subst(x, ts, interner)), Box::new(ty2.subst(x, ts, interner))),
            Type::Product(ref ty1, ref ty2) =>
                Type::Product(Box::new(ty1.subst(x, ts, interner)), Box::new(ty2.subst(x, ts, interner))),
            Type::Sum(ref ty1, ref ty2) =>
                Type::Sum(Box::new(ty1.subst(x, ts, interner)), Box::new(ty2.subst(x, ts, interner))),
            Type::Later(d, ref ty) => {
                let d_subst = if let &ToSubst::Clock(ref c) = ts {
                    d.substitute(x, c)
                } else { d };
                Type::Later(d_subst, Box::new(ty.subst(x, ts, interner)))
            },
            Type::Array(ref ty, ref size) =>
                Type::Array(Box::new(ty.subst(x, ts, interner)), size.clone()),
            Type::Box(ref ty) =>
                Type::Box(Box::new(ty.subst(x, ts, interner))),
            Type::Forall(y, k, ref ty) => {
                if x == y {
                    let new_name = mk_fresh(y, interner);
                    let replacement = match k {
                        Kind::Clock => ToSubst::Clock(Clock::from_var(new_name)),
                        Kind::Type => ToSubst::Type(Type::TypeVar(new_name)),
                    };
                    let freshened = ty.subst(y, &replacement, interner);
                    Type::Forall(new_name, k, Box::new(freshened.subst(x, ts, interner)))
                } else {
                    Type::Forall(y, k, Box::new(ty.subst(x, ts, interner)))
                }
            },
            Type::TypeVar(y) =>
                match *ts {
                    ToSubst::Type(ref ty) if x == y => ty.clone(),
                    _ => Type::TypeVar(y),
                },
            Type::Exists(y, ref ty) => {
                if x == y {
                    let new_name = mk_fresh(y, interner);
                    let replacement = ToSubst::Clock(Clock::from_var(new_name));
                    let freshened = ty.subst(y, &replacement, interner);
                    Type::Exists(new_name, Box::new(freshened.subst(x, ts, interner)))
                } else {
                    Type::Exists(y, Box::new(ty.subst(x, ts, interner)))
                }
            },
        }
    }

    // TODO: return a better error type
    fn check_validity(&self, ctx: &Ctx) -> Result<(), Symbol> {
        match *self {
            Type::Unit |
            Type::Sample |
            Type::Index =>
                Ok(()),
            Type::Stream(c, ref ty) |
            Type::Later(c, ref ty) =>
                if ctx.lookup_type_var(c.var) == Some(Kind::Clock) {
                    ty.check_validity(ctx)
                } else {
                    Err(c.var)
                },
            Type::Function(ref ty1, ref ty2) |
            Type::Product(ref ty1, ref ty2) |
            Type::Sum(ref ty1, ref ty2) => {
                ty1.check_validity(ctx)?;
                ty2.check_validity(ctx)
            },
            Type::Array(ref ty, _) |
            Type::Box(ref ty) =>
                ty.check_validity(ctx),
            Type::Forall(x, k, ref ty) => {
                let new_ctx = Ctx::TypeVar(x, k, Rc::new(ctx.clone()));
                ty.check_validity(&new_ctx)
            },
            Type::TypeVar(x) =>
                if ctx.lookup_type_var(x) == Some(Kind::Type) {
                    Ok(())
                } else {
                    Err(x)
                },
            Type::Exists(x, ref ty) => {
                let new_ctx = Ctx::TypeVar(x, Kind::Clock, Rc::new(ctx.clone()));
                ty.check_validity(&new_ctx)
            },
        }
    }
}

pub struct PrettyType<'a> {
    interner: &'a DefaultStringInterner,
    ty: &'a Type,
}

impl<'a> PrettyType<'a> {
    fn for_type(&self, ty: &'a Type) -> PrettyType<'a> {
        PrettyType { ty, ..*self }
    }

    fn for_clock(&self, clock: &'a Clock) -> PrettyClock<'a> {
        PrettyClock { clock, interner: self.interner }
    }
}

impl<'a> PrettyType<'a> {
    fn fmt_prec(&self, f: &mut fmt::Formatter<'_>, prec: u8) -> fmt::Result {
        match *self.ty {
            Type::Unit =>
                write!(f, "unit"),
            Type::Sample =>
                write!(f, "sample"),
            Type::Index =>
                write!(f, "index"),
            Type::Stream(ref clock, ref ty) =>
                parenthesize(f, prec > 3, |f| {
                    // oh GOD this syntax will be noisy
                    write!(f, "~^({}) ", self.for_clock(clock))?;
                    self.for_type(ty).fmt_prec(f, 3)
                }),
            Type::Function(ref ty1, ref ty2) =>
                parenthesize(f, prec > 0, |f| {
                    self.for_type(ty1).fmt_prec(f, 1)?;
                    write!(f, " -> ")?;
                    self.for_type(ty2).fmt_prec(f, 0)
                }),
            Type::Product(ref ty1, ref ty2) =>
                parenthesize(f, prec > 2, |f| {
                    self.for_type(ty1).fmt_prec(f, 3)?;
                    write!(f, " * ")?;
                    self.for_type(ty2).fmt_prec(f, 2)
                }),
            Type::Sum(ref ty1, ref ty2) =>
                parenthesize(f, prec > 1, |f| {
                    self.for_type(ty1).fmt_prec(f, 2)?;
                    write!(f, " * ")?;
                    self.for_type(ty2).fmt_prec(f, 1)
                }),
            Type::Later(ref clock, ref ty) =>
                parenthesize(f, prec > 3, |f| {
                    write!(f, "|>^({}) ", self.for_clock(clock))?;
                    self.for_type(ty).fmt_prec(f, 3)
                }),
            Type::Array(ref ty, ref size) => {
                write!(f, "[")?;
                self.for_type(ty).fmt_prec(f, 0)?;
                write!(f, "; {}]", size)
            },
            Type::Box(ref ty) =>
                parenthesize(f, prec > 3, |f| {
                    write!(f, "[] ")?;
                    self.for_type(ty).fmt_prec(f, 3)
                }),
            Type::Forall(x, k, ref ty) =>
                parenthesize(f, prec > 0, |f| {
                    write!(f, "for {} : {}. ", self.interner.resolve(x).unwrap(), k)?;
                    self.for_type(ty).fmt_prec(f, 0)
                }),
            Type::TypeVar(x) =>
                write!(f, "{}", self.interner.resolve(x).unwrap()),
            Type::Exists(x, ref ty) =>
                parenthesize(f, prec > 0, |f| {
                    write!(f, "? {}. ", self.interner.resolve(x).unwrap())?;
                    self.for_type(ty).fmt_prec(f, 0)
                }),
        }
    }
}

impl<'a> fmt::Display for PrettyType<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_prec(f, 0)
    }
}

// these should implicitly have boxes on them, i think? or, at least,
// they can always be used
pub type Globals = HashMap<Symbol, Type>;

// TODO: should probably find a more efficient representation of this,
// but it'll work for now
//
// TODO: want to keep variables around that are in lexical scope but
// removed due to tick-stripping, so that type errors can make more
// sense. but can't immediately figure out a nice way of doing that,
// so using this representation in the meantime.
#[derive(Clone, Debug)]
pub enum Ctx {
    Empty,
    Tick(Clock, Rc<Ctx>),
    TermVar(Symbol, Type, Rc<Ctx>),
    // variables behind this have "free" timing -- used as part of type synthesis
    Pretend(Rc<Ctx>),
    TypeVar(Symbol, Kind, Rc<Ctx>),
}

impl Ctx {
    fn lookup_term_var(&self, x: Symbol) -> Option<(Vec<Clock>, &Type)> {
        match *self {
            Ctx::Empty => None,
            Ctx::Tick(c, ref next) =>
                next.lookup_term_var(x).map(|(mut cs, ty)| {
                    cs.push(c);
                    (cs, ty)
                }),
            Ctx::TermVar(y, ref ty, ref next) =>
                if x == y {
                    Some((Vec::new(), ty))
                } else {
                    next.lookup_term_var(x)
                },
            Ctx::Pretend(ref next) =>
                next.lookup_term_var(x).map(|(_, ty)| (Vec::new(), ty)),
            Ctx::TypeVar(_, _, ref next) =>
                next.lookup_term_var(x),
        }
    }

    fn lookup_type_var(&self, x: Symbol) -> Option<Kind> {
        match *self {
            Ctx::Empty => None,
            Ctx::Tick(_, ref next) =>
                next.lookup_type_var(x),
            Ctx::TermVar(_, _, ref next) =>
                next.lookup_type_var(x),
            Ctx::Pretend(ref next) =>
                next.lookup_type_var(x),
            Ctx::TypeVar(y, k, ref next) =>
                if x == y {
                    Some(k)
                } else {
                    next.lookup_type_var(x)
                },
        }
    }

    fn with_var(self, x: Symbol, ty: Type) -> Ctx {
        Ctx::TermVar(x, ty, Rc::new(self))
    }

    fn with_type_var(self, x: Symbol, k: Kind) -> Ctx {
        Ctx::TypeVar(x, k, Rc::new(self))
    }

    // TODO: optimize this for the case that it's kept the same?
    fn box_strengthen(&self) -> Ctx {
        match *self {
            Ctx::Empty => Ctx::Empty,
            Ctx::Tick(_, ref next) => next.box_strengthen(),
            Ctx::TermVar(x, ref ty, ref next) =>
                if ty.is_stable() {
                    Ctx::TermVar(x, ty.clone(), Rc::new(next.box_strengthen()))
                } else {
                    next.box_strengthen()
                },
            Ctx::Pretend(ref next) =>
                Ctx::Pretend(Rc::new(next.box_strengthen())),
            Ctx::TypeVar(x, k, ref next) =>
                Ctx::TypeVar(x, k, Rc::new(next.box_strengthen())),
        }
    }

    fn strip_tick(&self, to_strip: Clock) -> Option<Ctx> {
        match *self {
            Ctx::Empty => None,
            Ctx::Tick(tick_amount, ref next) =>
                // TODO: is this sound????????
                match to_strip.partial_cmp(&tick_amount) {
                    Some(Ordering::Less) => {
                        let remaining_to_strip = to_strip.uncompose(&tick_amount).unwrap();
                        next.strip_tick(remaining_to_strip)
                    },
                    Some(Ordering::Equal) =>
                        Some((**next).clone()),
                    Some(Ordering::Greater) => {
                        let remaining_on_ctx = tick_amount.uncompose(&to_strip).unwrap();
                        Some(Ctx::Tick(remaining_on_ctx, next.clone()))
                    },
                    None =>
                        None
                },
            Ctx::TermVar(_, _, ref next) =>
                next.strip_tick(to_strip),
            Ctx::Pretend(ref _next) =>
                // uhhhhhhhhhh
                panic!("don't know what to do when trying to do tick-stripping in a pretend context!"),
            Ctx::TypeVar(x, _, ref next) =>
                if to_strip.var == x {
                    None
                } else {
                    next.strip_tick(to_strip)
                },
        }
    }

    fn pretty<'a>(&'a self, interner: &'a DefaultStringInterner) -> PrettyCtx<'a> {
        PrettyCtx { interner, ctx: self }
    }
}

pub struct PrettyCtx<'a> {
    interner: &'a DefaultStringInterner,
    ctx: &'a Ctx,
}

impl<'a> PrettyCtx<'a> {
    fn for_ctx(&self, ctx: &'a Ctx) -> PrettyCtx<'a> {
        PrettyCtx { ctx, ..*self }
    }
}

impl<'a> fmt::Display for PrettyCtx<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.ctx {
            Ctx::Empty =>
                write!(f, "-"),
            Ctx::TermVar(x, ref ty, ref next) =>
                write!(f, "{}, {}: {}",
                       self.for_ctx(next), self.interner.resolve(x).unwrap(),
                       PrettyType { interner: self.interner, ty }),
            Ctx::Tick(ref clock, ref next) =>
                write!(f, "{}, $^({})", self.for_ctx(next),
                       PrettyClock { interner: self.interner, clock }),
            Ctx::Pretend(ref next) =>
                write!(f, "{}, XX", self.for_ctx(next)),
            Ctx::TypeVar(x, k, ref next) =>
                write!(f, "{}, {} : {}", self.for_ctx(next), self.interner.resolve(x).unwrap(), k),
        }
    }
}

#[derive(Debug)]
pub enum TypeError<'a, R> {
    MismatchingTypes { expr: &'a Expr<'a, R>, synth: Type, expected: Type },
    VariableNotFound { range: R, var: Symbol },
    BadArgument { range: R, arg_type: Type, fun: &'a Expr<'a, R>, arg: &'a Expr<'a, R>, arg_err: Box<TypeError<'a, R>> },
    NonFunctionApplication { range: R, purported_fun: &'a Expr<'a, R>, actual_type: Type },
    SynthesisUnsupported { expr: &'a Expr<'a, R> },
    BadAnnotation { range: R, expr: &'a Expr<'a, R>, purported_type: Type, err: Box<TypeError<'a, R>> },
    LetSynthFailure { range: R, var: Symbol, expr: &'a Expr<'a, R>, err: Box<TypeError<'a, R>> },
    LetCheckFailure { range: R, var: Symbol, expected_type: Type, expr: &'a Expr<'a, R>, err: Box<TypeError<'a, R>> },
    ForcingNonThunk { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    UnPairingNonProduct { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    CasingNonSum { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    CouldNotUnify { type1: Type, type2: Type },
    MismatchingArraySize { range: R, expected_size: ArraySize, found_size: usize },
    UnGenningNonStream { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    VariableTimingBad { range: R, var: Symbol, timing: Vec<Clock>, var_type: Type },
    ForcingWithNotEnoughTick { range: R, expr: &'a Expr<'a, R>, ctx: Ctx, synthesized_clock: Clock },
    ForcingDoesntHoldUp { range: R, expr: &'a Expr<'a, R>, synthesized_clock: Clock, stripped_ctx: Ctx, err: Box<TypeError<'a, R>> },
    UnboxingNonBox { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    CouldntCheck { expr: &'a Expr<'a, R>, expected_type: Type, synthesis_error: Option<Box<TypeError<'a, R>>> },
    NonForallClockApp { range: R, purported_forall_clock: &'a Expr<'a, R>, actual_type: Type },
    NonForallTypeApp { range: R, purported_forall_type: &'a Expr<'a, R>, actual_type: Type },
    InvalidType { range: R, purported_type: Type, bad_symbol: Symbol },
    InvalidClock { range: R, purported_clock: Clock, bad_symbol: Symbol },
    ExElimNonExists { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
}

impl<'a, R> TypeError<'a, R> {
    fn mismatching(expr: &'a Expr<'a, R>, synth: Type, expected: Type) -> TypeError<'a, R> {
        TypeError::MismatchingTypes { expr, synth, expected }
    }

    fn var_not_found(range: R, var: Symbol) -> TypeError<'a, R> {
        TypeError::VariableNotFound { range, var }
    }

    fn bad_argument(range: R, arg_type: Type, fun: &'a Expr<'a, R>, arg: &'a Expr<'a, R>, arg_err: TypeError<'a, R>) -> TypeError<'a, R> {
        TypeError::BadArgument { range, arg_type, fun, arg, arg_err: Box::new(arg_err) }
    }

    fn non_function_application(range: R, purported_fun: &'a Expr<'a, R>, actual_type: Type) -> TypeError<'a, R> {
        TypeError::NonFunctionApplication { range, purported_fun, actual_type }
    }

    fn synthesis_unsupported(expr: &'a Expr<'a, R>) -> TypeError<'a, R> {
        TypeError::SynthesisUnsupported { expr }
    }

    fn bad_annotation(range: R, expr: &'a Expr<'a, R>, purported_type: Type, err: TypeError<'a, R>) -> TypeError<'a, R> {
        TypeError::BadAnnotation { range, expr, purported_type, err: Box::new(err) }
    }

    fn let_failure(range: R, var: Symbol, expr: &'a Expr<'a, R>, err: TypeError<'a, R>) -> TypeError<'a, R> {
        TypeError::LetSynthFailure { range, var, expr, err: Box::new(err) }
    }

    fn forcing_non_thunk(range: R, expr: &'a Expr<'a, R>, actual_type: Type) -> TypeError<'a, R> {
        TypeError::ForcingNonThunk { range, expr, actual_type }
    }

    fn unpairing_non_product(range: R, expr: &'a Expr<'a, R>, actual_type: Type) -> TypeError<'a, R> {
        TypeError::UnPairingNonProduct { range, expr, actual_type }
    }

    fn casing_non_sum(range: R, expr: &'a Expr<'a, R>, actual_type: Type) -> TypeError<'a, R> {
        TypeError::CasingNonSum { range, expr, actual_type }
    }

    fn could_not_unify(type1: Type, type2: Type) -> TypeError<'a, R> {
        TypeError::CouldNotUnify { type1, type2 }
    }

    pub fn pretty(&'a self, interner: &'a DefaultStringInterner, program_text: &'a str) -> PrettyTypeError<'a, R> {
        PrettyTypeError { interner, program_text, error: self }
    }

    fn source_type_error(&self) -> Option<&TypeError<'a, R>> {
        use TypeError::*;
        match *self {
            BadArgument { arg_err: ref err, .. } |
            BadAnnotation { ref err, .. } |
            LetSynthFailure { ref err, .. } |
            LetCheckFailure { ref err, .. } |
            CouldntCheck { synthesis_error: Some(ref err), .. } |
            ForcingDoesntHoldUp { ref err, .. } =>
                Some(err),
            _ =>
                None,
        }
    }
}

pub struct PrettyTypeError<'a, R> {
    interner: &'a DefaultStringInterner,
    program_text: &'a str,
    error: &'a TypeError<'a, R>,
}

impl<'a, R> PrettyTypeError<'a, R> {
    fn for_error(&self, error: &'a TypeError<'a, R>) -> PrettyTypeError<'a, R> {
        PrettyTypeError { interner: self.interner, program_text: self.program_text, error }
    }

    fn for_expr(&self, expr: &'a Expr<'a, tree_sitter::Range>) -> &'a str {
        let range = expr.range();
        &self.program_text[range.start_byte .. range.end_byte]
    }

    fn for_type(&self, ty: &'a Type) -> PrettyType<'a> {
        ty.pretty(self.interner)
    }

    fn for_timing(&self, timing: &'a [Clock]) -> PrettyTiming<'a> {
        PrettyTiming { interner: self.interner, timing }
    }

    fn for_clock(&self, clock: &'a Clock) -> PrettyClock<'a> {
        PrettyClock { interner: self.interner, clock }
    }

    fn for_ctx(&self, ctx: &'a Ctx) -> PrettyCtx<'a> {
        ctx.pretty(self.interner)
    }
}

impl<'a> PrettyTypeError<'a, tree_sitter::Range> {
    fn format_one(&self, f: &mut Indented<'_, fmt::Formatter<'_>>) -> fmt::Result {
        match *self.error {
            TypeError::MismatchingTypes { expr, ref synth, ref expected } =>
                write!(f, "found \"{}\" to have type \"{}\" but expected \"{}\"", self.for_expr(expr), self.for_type(synth), self.for_type(expected)),
            TypeError::VariableNotFound { var, .. } =>
                write!(f, "variable \"{}\" not found", self.interner.resolve(var).unwrap()),
            TypeError::BadArgument { ref arg_type, fun, arg, .. } =>
                write!(f, "found \"{}\" to take argument type \"{}\", but argument \"{}\" does not have that type",
                       self.for_expr(fun), self.for_type(arg_type), self.for_expr(arg)),
            TypeError::NonFunctionApplication { purported_fun, ref actual_type, .. } =>
                write!(f, "trying to call \"{}\", but found it to have type \"{}\", which is not a function type",
                       self.for_expr(purported_fun), self.for_type(actual_type)),
            TypeError::SynthesisUnsupported { expr } =>
                write!(f, "don't know how to implement synthesis for \"{}\" yet", self.for_expr(expr)),
            TypeError::BadAnnotation { expr, ref purported_type, .. } =>
                write!(f, "bad annotation of expression \"{}\" as type \"{}\"", self.for_expr(expr), self.for_type(purported_type)),
            TypeError::LetCheckFailure { var, expr, ref expected_type, .. } =>
                write!(f, "couldn't check variable \"{}\" to have type \"{}\" from definition \"{}\"",
                       self.interner.resolve(var).unwrap(), self.for_type(expected_type), self.for_expr(expr)),
            TypeError::LetSynthFailure { var, expr, .. } =>
                write!(f, "couldn't infer the type of variable \"{}\" from definition \"{}\"",
                       self.interner.resolve(var).unwrap(), self.for_expr(expr)),
            TypeError::ForcingNonThunk { expr, ref actual_type, .. } =>
                write!(f, "tried to force expression \"{}\" of type \"{}\", which is not a thunk", self.for_expr(expr), self.for_type(actual_type)),
            TypeError::UnPairingNonProduct { expr, ref actual_type, .. } =>
                write!(f, "tried to unpair expression \"{}\" of type \"{}\", which is not a product", self.for_expr(expr), self.for_type(actual_type)),
            TypeError::CasingNonSum { expr, ref actual_type, .. } =>
                write!(f, "tried to case on expression \"{}\" of type \"{}\", which is not a sum", self.for_expr(expr), self.for_type(actual_type)),
            TypeError::CouldNotUnify { ref type1, ref type2 } =>
                write!(f, "could not unify types \"{}\" and \"{}\"", self.for_type(type1), self.for_type(type2)),
            TypeError::MismatchingArraySize { ref expected_size, found_size, .. } =>
                write!(f, "expected array of size {} but found size {}", expected_size, found_size),
            TypeError::UnGenningNonStream { expr, ref actual_type, .. } =>
                write!(f, "expected stream to ungen, but found \"{}\" of type \"{}\"", self.for_expr(expr), self.for_type(actual_type)),
            TypeError::VariableTimingBad { var, ref timing, ref var_type, .. } =>
                write!(f, "found use of variable \"{}\", but it has timing {} and non-stable type \"{}\"",
                       self.interner.resolve(var).unwrap(), self.for_timing(timing), self.for_type(var_type)),
            TypeError::ForcingWithNotEnoughTick { expr, ref synthesized_clock, ref ctx, .. } =>
                write!(f, "trying to force expression \"{}\", but there is not enough tick for clock \"{}\" in the context \"{}\"",
                       self.for_expr(expr), self.for_clock(synthesized_clock), self.for_ctx(ctx)),
            TypeError::ForcingDoesntHoldUp { expr, ref stripped_ctx, .. } =>
                write!(f, "trying to force expression \"{}\", when the context has been stripped to \"{}\", it no longer typechecks!",
                       self.for_expr(expr), self.for_ctx(stripped_ctx)),
            TypeError::UnboxingNonBox { expr, ref actual_type, .. } =>
                write!(f, "trying to unbox expression \"{}\", but found it has type \"{}\", which is not a box",
                       self.for_expr(expr), self.for_type(actual_type)),
            TypeError::CouldntCheck { expr, ref expected_type, synthesis_error: Some(_) } =>
                write!(f, "expected \"{}\" to have type \"{}\", but couldn't check that and errored trying to synthesize a type",
                       self.for_expr(expr), self.for_type(expected_type)),
            TypeError::CouldntCheck { expr, ref expected_type, synthesis_error: None } =>
                write!(f, "expected \"{}\" to have type \"{}\", but couldn't check that!",
                       self.for_expr(expr), self.for_type(expected_type)),
            TypeError::NonForallClockApp { purported_forall_clock, ref actual_type, .. } =>
                write!(f, "can only apply clocks to forall-clock types, but expression \"{}\" has type \"{}\"",
                       self.for_expr(purported_forall_clock), self.for_type(actual_type)),
            TypeError::NonForallTypeApp { purported_forall_type, ref actual_type, .. } =>
                write!(f, "can only apply types to forall-type types, but expression \"{}\" has type \"{}\"",
                       self.for_expr(purported_forall_type), self.for_type(actual_type)),
            TypeError::InvalidType { ref purported_type, bad_symbol, .. } =>
                write!(f, "invalid type \"{}\"; could not find \"{}\" in the context",
                       self.for_type(purported_type), self.interner.resolve(bad_symbol).unwrap()),
            TypeError::InvalidClock { ref purported_clock, bad_symbol, .. } =>
                write!(f, "invalid clock \"{}\"; could not find \"{}\" as a clock in the context",
                       self.for_clock(purported_clock), self.interner.resolve(bad_symbol).unwrap()),
            TypeError::ExElimNonExists { expr, ref actual_type, .. } =>
                write!(f, "trying to exists-elim expression \"{}\" of type \"{}\", which is not an exists",
                       self.for_expr(expr), self.for_type(actual_type)),
        }
    }
}

impl<'a> fmt::Display for PrettyTypeError<'a, tree_sitter::Range> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut error = Some(self.error);
        let mut indent: usize = 0;

        while let Some(err) = error {
            let mut inserter = move |_: usize, f: &mut dyn Write|
              (0..indent).map(|_| write!(f, "  ")).collect();
            self.for_error(err).format_one(&mut indented(f).with_format(Format::Custom {
                inserter: &mut inserter
            }))?;
            writeln!(f)?;

            error = err.source_type_error();
            indent += 1;
        }

        Ok(())
    }
}

pub struct Typechecker<'a> {
    pub globals: &'a mut Globals,
    pub interner: &'a mut DefaultStringInterner,
}


// rules to port/verify:
// - [X] delay
// - [X] adv
// - [X] unbox
// - [X] box
// - [X] gen
// - [X] ungen
// - [ ] proj?
// - [X] fix

impl<'a> Typechecker<'a> {
    pub fn check<'b, R: Clone>(&mut self, ctx: &Ctx, expr: &'b Expr<'b, R>, ty: &Type) -> Result<(), TypeError<'b, R>> {
        match (ty, expr) {
            // CAREFUL! positioning this here matters, with respect to
            // the other non-expression-syntax-guided rules
            // (particlarly lob)
            (&Type::Forall(x, k, ref ty), _) =>
                self.check(&Ctx::TypeVar(x, k, Rc::new(ctx.clone())), expr, ty),
            (&Type::Unit, &Expr::Val(_, Value::Unit)) =>
                Ok(()),
            (&Type::Function(ref ty1, ref ty2), &Expr::Lam(_, x, e)) => {
                let new_ctx = ctx.clone().with_var(x, (**ty1).clone());
                self.check(&new_ctx, e, ty2)
            },
            (_, &Expr::Lob(_, clock, x, e)) => {
                let rec_ty = Type::Box(Box::new(Type::Later(clock, Box::new(ty.clone()))));
                let new_ctx = ctx.box_strengthen().with_var(x, rec_ty);
                self.check(&new_ctx, e, ty)
            },
            // if we think of streams as infinitary products, it makes sense to *check* their introduction, right?
            (&Type::Stream(clock, ref ty1), &Expr::Gen(_, eh, et)) => {
                // TODO: probably change once we figure out the stream semantics we actually want
                self.check(&ctx, eh, ty1)?;
                self.check(&ctx, et, &Type::Later(clock, Box::new(ty.clone())))
            },
            (_, &Expr::LetIn(ref r, x, None, e1, e2)) =>
                match self.synthesize(ctx, e1) {
                    Ok(ty_x) => {
                        let new_ctx = ctx.clone().with_var(x, ty_x);
                        self.check(&new_ctx, e2, ty)
                    },
                    Err(err) =>
                        Err(TypeError::let_failure(r.clone(), x, e1, err)),
                },
            (_, &Expr::LetIn(ref r, x, Some(ref e1_ty), e1, e2)) => {
                if let Err(err) = self.check(ctx, e1, e1_ty) {
                    return Err(TypeError::LetCheckFailure {
                        range: r.clone(),
                        var: x,
                        expected_type: e1_ty.clone(),
                        expr: e1,
                        err: Box::new(err)
                    });
                }
                let new_ctx = ctx.clone().with_var(x, e1_ty.clone());
                self.check(&new_ctx, e2, ty)
            },
            (&Type::Product(ref ty1, ref ty2), &Expr::Pair(_, e1, e2)) => {
                self.check(ctx, e1, ty1)?;
                self.check(ctx, e2, ty2)
            },
            (_, &Expr::UnPair(ref r, x1, x2, e0, e)) =>
                match self.synthesize(ctx, e0)? {
                    Type::Product(ty1, ty2) => {
                        let new_ctx = ctx.clone().with_var(x1, *ty1).with_var(x2, *ty2);
                        self.check(&new_ctx, e, ty)
                    },
                    ty =>
                        Err(TypeError::unpairing_non_product(r.clone(), e0, ty)),
                },
            (_, &Expr::ExElim(ref r, c, x, e0, e)) =>
                match self.synthesize(ctx, e0)? {
                    Type::Exists(d, ty_var) => {
                        let ty_substed = ty_var.subst(d, &ToSubst::Clock(Clock::from_var(c)), self.interner);
                        let new_ctx = ctx.clone().with_type_var(c, Kind::Clock).with_var(x, ty_substed);
                        self.check(&new_ctx, e, ty)
                    },
                    ty =>
                        Err(TypeError::ExElimNonExists { range: r.clone(), expr: e0, actual_type: ty }),
                },
            (&Type::Sum(ref ty1, _), &Expr::InL(_, e)) =>
                self.check(ctx, e, ty1),
            (&Type::Sum(_, ref ty2), &Expr::InR(_, e)) =>
                self.check(ctx, e, ty2),
            (_, &Expr::Case(ref r, e0, x1, e1, x2, e2)) =>
                match self.synthesize(ctx, e0)? {
                    Type::Sum(ty1, ty2) => {
                        let old_ctx = Rc::new(ctx.clone());
                        let ctx1 = Ctx::TermVar(x1, *ty1, old_ctx.clone());
                        self.check(&ctx1, e1, ty)?;
                        let ctx2 = Ctx::TermVar(x2, *ty2, old_ctx);
                        self.check(&ctx2, e2, ty)
                    },
                    ty =>
                        Err(TypeError::casing_non_sum(r.clone(), e0, ty)),
                },
            (&Type::Array(ref ty, ref size), &Expr::Array(ref r, ref es)) =>
                if size.as_const() != Some(es.len()) {
                    Err(TypeError::MismatchingArraySize {
                        range: r.clone(),
                        expected_size: size.clone(),
                        found_size: es.len()
                    })
                } else {
                    for e in es.iter() {
                        self.check(ctx, e, ty)?;
                    }
                    Ok(())
                },
            (&Type::Later(clock, ref ty), &Expr::Delay(_, e)) => {
                let new_ctx = Ctx::Tick(clock, Rc::new(ctx.clone()));
                self.check(&new_ctx, e, ty)
            },
            (&Type::Box(ref ty), &Expr::Box(_, e)) =>
                self.check(&ctx.box_strengthen(), e, ty),
            (&Type::Exists(c, ref ty), &Expr::ExIntro(ref r, d, e)) => {
                match ctx.lookup_type_var(d.var) {
                    Some(Kind::Clock) => { }
                    _ => {
                        return Err(TypeError::InvalidClock { range: r.clone(), purported_clock: d, bad_symbol: d.var });
                    }
                }
                let expected = ty.subst(c, &ToSubst::Clock(d), self.interner);
                self.check(ctx, e, &expected)
            },
            (_, _) => {
                let synthesized = match self.synthesize(ctx, expr) {
                    Ok(ty) =>
                        ty,
                    Err(TypeError::SynthesisUnsupported { .. }) =>
                        return Err(TypeError::CouldntCheck {
                            expr,
                            expected_type: ty.clone(),
                            synthesis_error: None,
                        }),
                    Err(err) =>
                        return Err(TypeError::CouldntCheck {
                            expr,
                            expected_type: ty.clone(),
                            synthesis_error: Some(Box::new(err)),
                        }),
                };

                if subtype(ctx, &synthesized, ty, self.interner) {
                    Ok(())
                } else {
                    Err(TypeError::mismatching(expr, synthesized, ty.clone()))
                }
            }
        }
    }

    pub fn synthesize<'b, R: Clone>(&mut self, ctx: &Ctx, expr: &'b Expr<'b, R>) -> Result<Type, TypeError<'b, R>> {
        match expr {
            &Expr::Val(_, ref v) =>
                match *v {
                    Value::Unit => Ok(Type::Unit),
                    Value::Sample(_) => Ok(Type::Sample),
                    Value::Index(_) => Ok(Type::Index),
                    _ => panic!("trying to type {v:?} but that kind of value shouldn't be created yet?"),
                }
            &Expr::Var(ref r, x) =>
                if let Some((timing, ty)) = ctx.lookup_term_var(x) {
                    if timing.is_empty() || ty.is_stable() {
                        Ok(ty.clone())
                    } else {
                        Err(TypeError::VariableTimingBad {
                            range: r.clone(),
                            var: x,
                            timing,
                            var_type: ty.clone(),
                        })
                    }
                } else if let Some(ty) = self.globals.get(&x) {
                    Ok(ty.clone())
                } else {
                    Err(TypeError::var_not_found(r.clone(), x))
                },
            &Expr::Annotate(ref r, e, ref ty) => {
                ty.check_validity(ctx).map_err(|bad_symbol|
                    TypeError::InvalidType {
                        range: r.clone(),
                        purported_type: ty.clone(),
                        bad_symbol
                    }
                )?;

                match self.check(ctx, e, ty) {
                    Ok(()) => Ok(ty.clone()),
                    Err(err) => Err(TypeError::bad_annotation(r.clone(), e, ty.clone(), err)),
                }
            },
            &Expr::App(ref r, e1, e2) =>
                match self.synthesize(ctx, e1)? {
                    Type::Function(ty_a, ty_b) => {
                        match self.check(ctx, e2, &ty_a) {
                            Ok(()) => Ok(*ty_b),
                            Err(arg_err) => Err(TypeError::bad_argument(r.clone(), *ty_a, e1, e2, arg_err)),
                        }
                    },
                    ty =>
                        Err(TypeError::non_function_application(r.clone(), e1, ty)),
                },
            &Expr::Adv(ref r, e1) => {
                // first we synthesize the type under the assumption
                // that we can use any variable freely, then we strip
                // the context according to what the type says we
                // must, and finally re-check/synthesize the type
                // under the stripped context. hacky but may work...
                //
                // TODO: need to be careful about variable shadowing
                // here... will probably need to move to a better
                // context representation
                let pretend = Ctx::Pretend(Rc::new(ctx.clone()));
                let (synthesized_clock, synthesized_type) = match self.synthesize(&pretend, e1)? {
                    Type::Later(clock, ty) => (clock, ty),
                    ty => return Err(TypeError::forcing_non_thunk(r.clone(), e1, ty)),
                };
                let Some(stripped_ctx) = ctx.strip_tick(synthesized_clock) else {
                    return Err(TypeError::ForcingWithNotEnoughTick {
                        range: r.clone(),
                        expr: e1,
                        ctx: ctx.clone(),
                        synthesized_clock,
                    });
                };
                match self.check(&stripped_ctx, e1, &Type::Later(synthesized_clock, synthesized_type.clone())) {
                    Ok(()) =>
                        Ok(*synthesized_type),
                    Err(err) =>
                        Err(TypeError::ForcingDoesntHoldUp {
                            range: r.clone(),
                            expr: e1,
                            synthesized_clock,
                            stripped_ctx,
                            err: Box::new(err),
                        }),
                }
            },
            &Expr::LetIn(ref r, x, None, e1, e2) =>
                match self.synthesize(ctx, e1) {
                    Ok(ty_x) => {
                        let new_ctx = ctx.clone().with_var(x, ty_x);
                        self.synthesize(&new_ctx, e2)
                    },
                    Err(err) =>
                        Err(TypeError::let_failure(r.clone(), x, e1, err)),
                },
            &Expr::LetIn(ref r, x, Some(ref e1_ty), e1, e2) => {
                if let Err(err) = self.check(ctx, e1, e1_ty) {
                    return Err(TypeError::LetCheckFailure {
                        range: r.clone(),
                        var: x,
                        expected_type: e1_ty.clone(),
                        expr: e1,
                        err: Box::new(err)
                    });
                }
                let new_ctx = ctx.clone().with_var(x, e1_ty.clone());
                self.synthesize(&new_ctx, e2)
            },
            &Expr::UnPair(ref r, x1, x2, e0, e) =>
                match self.synthesize(ctx, e0)? {
                    Type::Product(ty1, ty2) => {
                        let new_ctx = ctx.clone().with_var(x1, *ty1).with_var(x2, *ty2);
                        self.synthesize(&new_ctx, e)
                    },
                    ty =>
                        Err(TypeError::unpairing_non_product(r.clone(), e0, ty)),
                },
            &Expr::ExElim(ref r, c, x, e0, e) =>
                match self.synthesize(ctx, e0)? {
                    Type::Exists(d, ty) => {
                        let ty_substed = ty.subst(d, &ToSubst::Clock(Clock::from_var(c)), self.interner);
                        let new_ctx = ctx.clone().with_type_var(c, Kind::Clock).with_var(x, ty_substed);
                        self.synthesize(&new_ctx, e)
                    },
                    ty =>
                        Err(TypeError::ExElimNonExists { range: r.clone(), expr: e0, actual_type: ty }),
                },
            &Expr::Case(ref r, e0, x1, e1, x2, e2) =>
                match self.synthesize(ctx, e0)? {
                    Type::Sum(ty1, ty2) => {
                        let old_ctx = Rc::new(ctx.clone());
    
                        let ctx1 = Ctx::TermVar(x1, *ty1, old_ctx.clone());
                        let ty_out1 = self.synthesize(&ctx1, e1)?;
    
                        let ctx2 = Ctx::TermVar(x2, *ty2, old_ctx);
                        let ty_out2 = self.synthesize(&ctx2, e2)?;
    
                        meet(ctx, ty_out1, ty_out2, self.interner)
                    },
                    ty =>
                        Err(TypeError::casing_non_sum(r.clone(), e0, ty)),
                },
            &Expr::UnGen(ref r, e) =>
                match self.synthesize(ctx, e)? {
                    Type::Stream(clock, ty) =>
                        Ok(Type::Product(ty.clone(), Box::new(Type::Later(clock, Box::new(Type::Stream(clock, ty)))))),
                    ty =>
                        Err(TypeError::UnGenningNonStream { range: r.clone(), expr: e, actual_type: ty }),
                }
            &Expr::Unbox(ref r, e) =>
                match self.synthesize(ctx, e)? {
                    Type::Box(ty) =>
                        Ok(*ty),
                    ty =>
                        Err(TypeError::UnboxingNonBox { range: r.clone(), expr: e, actual_type: ty }),
                },
            &Expr::ClockApp(ref r, e, c) =>
                // TODO: check validity of c
                match self.synthesize(ctx, e)? {
                    Type::Forall(x, Kind::Clock, ty) =>
                        Ok(ty.subst(x, &ToSubst::Clock(c), self.interner)),
                    ty =>
                        Err(TypeError::NonForallClockApp { range: r.clone(), purported_forall_clock: e, actual_type: ty }),
                },
            &Expr::TypeApp(ref r, e, ref ty_to_subst) => {
                ty_to_subst.check_validity(ctx).map_err(|bad_symbol|
                    TypeError::InvalidType {
                        range: r.clone(),
                        purported_type: ty_to_subst.clone(),
                        bad_symbol
                    }
                )?;

                match self.synthesize(ctx, e)? {
                    Type::Forall(x, Kind::Type, ty) =>
                        Ok(ty.subst(x, &ToSubst::Type(ty_to_subst.clone()), self.interner)),
                    ty =>
                        Err(TypeError::NonForallTypeApp { range: r.clone(), purported_forall_type: e, actual_type: ty }),
                }
            },
            &Expr::Binop(ref _r, op, e1, e2) => {
                // TODO: make this const somewhere and somehow
                let tybool = Type::Sum(Box::new(Type::Unit), Box::new(Type::Unit));
                let (ty1, ty2, tyret) = match op {
                    Binop::FMul => (Type::Sample, Type::Sample, Type::Sample),
                    Binop::FDiv => (Type::Sample, Type::Sample, Type::Sample),
                    Binop::FAdd => (Type::Sample, Type::Sample, Type::Sample),
                    Binop::FSub => (Type::Sample, Type::Sample, Type::Sample),
                    Binop::FGt => (Type::Sample, Type::Sample, tybool),
                    Binop::FGe => (Type::Sample, Type::Sample, tybool),
                    Binop::FLt => (Type::Sample, Type::Sample, tybool),
                    Binop::FLe => (Type::Sample, Type::Sample, tybool),
                    Binop::FEq => (Type::Sample, Type::Sample, tybool),
                    Binop::FNe => (Type::Sample, Type::Sample, tybool),
                    Binop::Shl => (Type::Index, Type::Index, Type::Index),
                    Binop::Shr => (Type::Index, Type::Index, Type::Index),
                    Binop::And => (Type::Index, Type::Index, Type::Index),
                    Binop::Xor => (Type::Index, Type::Index, Type::Index),
                    Binop::Or => (Type::Index, Type::Index, Type::Index),
                    Binop::IMul => (Type::Index, Type::Index, Type::Index),
                    Binop::IDiv => (Type::Index, Type::Index, Type::Index),
                    Binop::IAdd => (Type::Index, Type::Index, Type::Index),
                    Binop::ISub => (Type::Index, Type::Index, Type::Index),
                    Binop::IGt => (Type::Index, Type::Index, tybool),
                    Binop::IGe => (Type::Index, Type::Index, tybool),
                    Binop::ILt => (Type::Index, Type::Index, tybool),
                    Binop::ILe => (Type::Index, Type::Index, tybool),
                    Binop::IEq => (Type::Index, Type::Index, tybool),
                    Binop::INe => (Type::Index, Type::Index, tybool),
                };
                // TODO: should probably have a more specific type error for this?
                self.check(ctx, e1, &ty1)?;
                self.check(ctx, e2, &ty2)?;
                Ok(tyret)
            },
            _ =>
                Err(TypeError::synthesis_unsupported(expr)),
        }
    }

    pub fn check_file<'c, 'b, R: Clone>(&mut self, file: &'c SourceFile<'b, R>) -> Result<(), FileTypeErrors<'b, R>> {
        let mut errs = Vec::new();
        let mut running_ctx = Ctx::Empty;
        for def in file.defs.iter() {
            let ctx = match def.kind {
                TopLevelDefKind::Let => &running_ctx,
                TopLevelDefKind::Def => &Ctx::Empty,
            };
            if let Err(err) = self.check(ctx, &def.body, &def.type_) {
                errs.push(TopLevelTypeError::TypeError(def.name, err));
            }
            if running_ctx.lookup_term_var(def.name).is_some() ||
                self.globals.get(&def.name).is_some() {
                errs.push(TopLevelTypeError::CannotRedefine(def.name, def.range.clone()));
            }
            match def.kind {
                TopLevelDefKind::Let => {
                    running_ctx = Ctx::TermVar(def.name, def.type_.clone(), Rc::new(running_ctx));
                },
                TopLevelDefKind::Def => {
                    self.globals.insert(def.name, def.type_.clone());
                },
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(FileTypeErrors { errs })
        }
    }
}

#[derive(Debug)]
pub enum TopLevelTypeError<'b, R> {
    TypeError(Symbol, TypeError<'b, R>),
    CannotRedefine(Symbol, R),
}

#[derive(Debug)]
pub struct FileTypeErrors<'b, R> {
    pub errs: Vec<TopLevelTypeError<'b, R>>,
}

impl<'b, R> FileTypeErrors<'b, R> {
    pub fn pretty(&'b self, interner: &'b DefaultStringInterner, code: &'b str) -> PrettyFileTypeErrors<'b, R> {
        PrettyFileTypeErrors { interner, code, errs: self }
    }
}

pub struct PrettyFileTypeErrors<'b, R> {
    interner: &'b DefaultStringInterner,
    code: &'b str,
    errs: &'b FileTypeErrors<'b, R>,
}

impl<'b> fmt::Display for PrettyFileTypeErrors<'b, tree_sitter::Range> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for err in self.errs.errs.iter() {
            match *err {
                TopLevelTypeError::TypeError(name, ref err) =>
                    write!(f, "in definition of \"{}\": {}",
                           self.interner.resolve(name).unwrap(),
                           err.pretty(self.interner, self.code))?,
                TopLevelTypeError::CannotRedefine(name, _) =>
                    write!(f, "cannot redefine \"{}\"", self.interner.resolve(name).unwrap())?,
            }
        }
        Ok(())
    }
}

// at the moment this implements an equivalence relation, but we'll
// probably want it to be proper subtyping at some point, so let's
// just call it that
//
// terminating: the sum of the sizes of the types decreases
fn subtype(ctx: &Ctx, ty1: &Type, ty2: &Type, interner: &mut DefaultStringInterner) -> bool {
    match (ty1, ty2) {
        (&Type::Unit, &Type::Unit) =>
            true,
        (&Type::Sample, &Type::Sample) =>
            true,
        (&Type::Index, &Type::Index) =>
            true,
        (&Type::Stream(ref c1, ref ty1p), &Type::Stream(ref c2, ref ty2p)) =>
            c1 == c2 && subtype(ctx, ty1p, ty2p, interner),
        (&Type::Function(ref ty1a, ref ty1b), &Type::Function(ref ty2a, ref ty2b)) =>
            subtype(ctx, ty2a, ty1a, interner) && subtype(ctx, ty1b, ty2b, interner),
        (&Type::Product(ref ty1a, ref ty1b), &Type::Product(ref ty2a, ref ty2b)) =>
            subtype(ctx, ty1a, ty2a, interner) && subtype(ctx, ty1b, ty2b, interner),
        (&Type::Sum(ref ty1a, ref ty1b), &Type::Sum(ref ty2a, ref ty2b)) =>
            subtype(ctx, ty1a, ty2a, interner) && subtype(ctx, ty1b, ty2b, interner),
        (&Type::Later(ref c1, ref ty1p), &Type::Later(ref c2, ref ty2p)) =>
            match c1.partial_cmp(c2) {
                Some(Ordering::Less) => {
                    // unwrap safety: we've already verified that c1 < c2
                    let rem = c1.uncompose(c2).unwrap();
                    subtype(ctx, &Type::Later(rem, ty1p.clone()), ty2p, interner)
                },
                Some(Ordering::Equal) =>
                    subtype(ctx, ty1p, ty2p, interner),
                Some(Ordering::Greater) => {
                    // unwrap safety: we've already verified that c2 < c1
                    let rem = c2.uncompose(c1).unwrap();
                    subtype(ctx, ty1p, &Type::Later(rem, ty2p.clone()), interner)
                },
                None =>
                    false,
            }
        (&Type::Array(ref ty1p, ref n1), &Type::Array(ref ty2p, ref n2)) =>
            subtype(ctx, ty1p, ty2p, interner) && n1 == n2,
        (&Type::Box(ref ty1p), &Type::Box(ref ty2p)) =>
            subtype(ctx, ty1p, ty2p, interner),
        (&Type::Forall(x1, k1, ref ty1p), &Type::Forall(x2, k2, ref ty2p)) if k1 == k2 => {
            let fresh_name = mk_fresh(x1, interner);
            let replacement = ToSubst::from_var(fresh_name, k1);
            let ty1p_subst = ty1p.subst(x1, &replacement, interner);
            let ty2p_subst = ty2p.subst(x2, &replacement, interner);
            let new_ctx = Ctx::TypeVar(fresh_name, k1, Rc::new(ctx.clone()));
            subtype(&new_ctx, &ty1p_subst, &ty2p_subst, interner)
        },
        (&Type::TypeVar(x1), &Type::TypeVar(x2)) =>
            x1 == x2,
        (&Type::Exists(x1, ref ty1p), &Type::Exists(x2, ref ty2p)) => {
            let fresh_name = mk_fresh(x1, interner);
            let replacement = ToSubst::from_var(fresh_name, Kind::Clock);
            let ty1p_subst = ty1p.subst(x1, &replacement, interner);
            let ty2p_subst = ty2p.subst(x2, &replacement, interner);
            let new_ctx = Ctx::TypeVar(fresh_name, Kind::Clock, Rc::new(ctx.clone()));
            subtype(&new_ctx, &ty1p_subst, &ty2p_subst, interner)
        },
        (_, _) =>
            false,
    }
}

fn meet<'a, R>(ctx: &Ctx, ty1: Type, ty2: Type, interner: &mut DefaultStringInterner) -> Result<Type, TypeError<'a, R>> {
    if subtype(ctx, &ty1, &ty2, interner) {
        Ok(ty2)
    } else if subtype(ctx, &ty2, &ty1, interner) {
        Ok(ty1)
    } else {
        Err(TypeError::could_not_unify(ty1, ty2))
    }
}
