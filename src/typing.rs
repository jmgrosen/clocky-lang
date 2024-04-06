use core::fmt;
use std::collections::HashMap;

use string_interner::DefaultStringInterner;

use crate::expr::{Symbol, Expr, Value, PrettyExpr};

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

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Type {
    Unit,
    Sample,
    Index,
    Stream(Box<Type>),
    Function(Box<Type>, Box<Type>),
    Product(Box<Type>, Box<Type>),
    Sum(Box<Type>, Box<Type>),
    Later(Box<Type>),
    Array(Box<Type>, ArraySize),
}

pub type Ctx = HashMap<Symbol, Type>;

#[derive(Debug)]
pub enum TypeError<'a, R> {
    MismatchingTypes { expr: &'a Expr<'a, R>, synth: Type, expected: Type },
    VariableNotFound { range: R, var: Symbol },
    BadArgument { range: R, arg_type: Type, fun: &'a Expr<'a, R>, arg: &'a Expr<'a, R>, arg_err: Box<TypeError<'a, R>> },
    NonFunctionApplication { range: R, purported_fun: &'a Expr<'a, R>, actual_type: Type },
    Unsupported { expr: &'a Expr<'a, R> },
    BadAnnotation { range: R, expr: &'a Expr<'a, R>, purported_type: Type, err: Box<TypeError<'a, R>> },
    LetFailure { range: R, var: Symbol, expr: &'a Expr<'a, R>, err: Box<TypeError<'a, R>> },
    ForcingNonThunk { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    UnPairingNonProduct { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    CasingNonSum { range: R, expr: &'a Expr<'a, R>, actual_type: Type },
    CouldNotUnify { type1: Type, type2: Type },
    MismatchingArraySize { range: R, expected_size: ArraySize, found_size: usize },
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

    fn unsupported(expr: &'a Expr<'a, R>) -> TypeError<'a, R> {
        TypeError::Unsupported { expr }
    }

    fn bad_annotation(range: R, expr: &'a Expr<'a, R>, purported_type: Type, err: TypeError<'a, R>) -> TypeError<'a, R> {
        TypeError::BadAnnotation { range, expr, purported_type, err: Box::new(err) }
    }

    fn let_failure(range: R, var: Symbol, expr: &'a Expr<'a, R>, err: TypeError<'a, R>) -> TypeError<'a, R> {
        TypeError::LetFailure { range, var, expr, err: Box::new(err) }
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

    fn for_expr(&self, expr: &'a Expr<'a, R>) -> PrettyExpr<'a, 'a, R> {
        expr.pretty(self.interner)
    }
}

impl<'a, R> fmt::Display for PrettyTypeError<'a, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.error {
            TypeError::MismatchingTypes { expr, ref synth, ref expected } =>
                write!(f, "found {} to have type {:?} but expected {:?}", self.for_expr(expr), synth, expected),
            TypeError::VariableNotFound { var, .. } =>
                write!(f, "variable \"{}\" not found", self.interner.resolve(var).unwrap()),
            TypeError::BadArgument { ref arg_type, fun, arg, ref arg_err, .. } =>
                write!(f, "found {} to take argument type {:?}, but argument {} does not have that type: {}",
                       self.for_expr(fun), arg_type, self.for_expr(arg), self.for_error(arg_err)),
            TypeError::NonFunctionApplication { purported_fun, ref actual_type, .. } =>
                write!(f, "trying to call {}, but found it to have type {:?}, which is not a function type",
                       self.for_expr(purported_fun), actual_type),
            TypeError::Unsupported { expr } =>
                write!(f, "oops haven't implemented typing rules for {} yet", self.for_expr(expr)),
            TypeError::BadAnnotation { expr, ref purported_type, ref err, .. } =>
                write!(f, "bad annotation of expression {} as type {:?}: {}", self.for_expr(expr), purported_type, self.for_error(err)),
            TypeError::LetFailure { var, expr, ref err, .. } =>
                write!(f, "couldn't infer the type of variable {} from definition {}: {}",
                       self.interner.resolve(var).unwrap(), self.for_expr(expr), self.for_error(err)),
            TypeError::ForcingNonThunk { expr, ref actual_type, .. } =>
                write!(f, "tried to force expression {} of type {:?}, which is not a thunk", self.for_expr(expr), actual_type),
            TypeError::UnPairingNonProduct { expr, ref actual_type, .. } =>
                write!(f, "tried to unpair expression {} of type {:?}, which is not a product", self.for_expr(expr), actual_type),
            TypeError::CasingNonSum { expr, ref actual_type, .. } =>
                write!(f, "tried to case on expression {} of type {:?}, which is not a sum", self.for_expr(expr), actual_type),
            TypeError::CouldNotUnify { ref type1, ref type2 } =>
                write!(f, "could not unify types {:?} and {:?}", type1, type2),
            TypeError::MismatchingArraySize { ref expected_size, found_size, .. } =>
                write!(f, "expected array of size {:?} but found size {}", expected_size, found_size),
        }
    }
}

pub fn check<'a, R: Clone>(ctx: &Ctx, expr: &'a Expr<'a, R>, ty: &Type) -> Result<(), TypeError<'a, R>> {
    match (ty, expr) {
        (&Type::Unit, &Expr::Val(_, Value::Unit)) =>
            Ok(()),
        (&Type::Function(ref ty1, ref ty2), &Expr::Lam(_, x, e)) => {
            let mut new_ctx = ctx.clone();
            new_ctx.insert(x, (**ty1).clone());
            check(&new_ctx, e, ty2)
        },
        (_, &Expr::Lob(_, x, e)) => {
            let mut new_ctx = ctx.clone();
            new_ctx.insert(x, Type::Later(Box::new(ty.clone())));
            check(&new_ctx, e, ty)
        },
        // if we think of streams as infinitary products, it makes sense to *check* their introduction, right?
        (&Type::Stream(ref ty1), &Expr::Gen(_, eh, et)) => {
            // TODO: probably change once we figure out the stream semantics we actually want
            check(&ctx, eh, ty1)?;
            check(&ctx, et, ty)
        },
        (_, &Expr::LetIn(ref r, x, e1, e2)) =>
            match synthesize(ctx, e1) {
                Ok(ty_x) => {
                    let mut new_ctx = ctx.clone();
                    new_ctx.insert(x, ty_x);
                    check(&new_ctx, e2, ty)
                },
                Err(err) =>
                    Err(TypeError::let_failure(r.clone(), x, e1, err)),
            },
        (&Type::Product(ref ty1, ref ty2), &Expr::Pair(_, e1, e2)) => {
            check(ctx, e1, ty1)?;
            check(ctx, e2, ty2)
        },
        (_, &Expr::UnPair(ref r, x1, x2, e0, e)) =>
            match synthesize(ctx, e0)? {
                Type::Product(ty1, ty2) => {
                    let mut new_ctx = ctx.clone();
                    new_ctx.insert(x1, *ty1);
                    new_ctx.insert(x2, *ty2);
                    check(&new_ctx, e, ty)
                },
                ty =>
                    Err(TypeError::unpairing_non_product(r.clone(), e0, ty)),
            },
        (&Type::Sum(ref ty1, _), &Expr::InL(_, e)) =>
            check(ctx, e, ty1),
        (&Type::Sum(_, ref ty2), &Expr::InR(_, e)) =>
            check(ctx, e, ty2),
        (_, &Expr::Case(ref r, e0, x1, e1, x2, e2)) =>
            match synthesize(ctx, e0)? {
                Type::Sum(ty1, ty2) => {
                    let mut ctx1 = ctx.clone();
                    ctx1.insert(x1, *ty1);
                    check(&ctx1, e1, ty)?;
                    let mut ctx2 = ctx.clone();
                    ctx2.insert(x2, *ty2);
                    check(&ctx2, e2, ty)
                },
                ty =>
                    Err(TypeError::casing_non_sum(r.clone(), e0, ty)),
            },
        (&Type::Array(ref ty, ref size), &Expr::Array(ref r, ref es)) =>
            if size.as_const() != Some(es.len()) {
                Err(TypeError::MismatchingArraySize { range: r.clone(), expected_size: size.clone(), found_size: es.len() })
            } else {
                for e in es.iter() {
                    check(ctx, e, ty)?;
                }
                Ok(())
            },
        (_, _) => {
            let synthesized = synthesize(ctx, expr)?;
            if synthesized == *ty {
                Ok(())
            } else {
                Err(TypeError::mismatching(expr, synthesized, ty.clone()))
            }
        }
    }
}

pub fn synthesize<'a, R: Clone>(ctx: &Ctx, expr: &'a Expr<'a, R>) -> Result<Type, TypeError<'a, R>> {
    match expr {
        &Expr::Val(_, ref v) =>
            match *v {
                Value::Unit => Ok(Type::Unit),
                Value::Sample(_) => Ok(Type::Sample),
                Value::Index(_) => Ok(Type::Index),
                Value::Gen(_, _, _) |
                Value::Closure(_, _, _) |
                Value::Suspend(_, _) |
                Value::Pair(_, _) |
                Value::InL(_) |
                Value::InR(_) |
                Value::Array(_) |
                Value::BuiltinPartial(_, _) =>
                    panic!("trying to type {v:?} but that kind of value shouldn't be created yet?"),
            }
        &Expr::Var(ref r, x) =>
            ctx.get(&x).cloned().ok_or_else(|| TypeError::var_not_found(r.clone(), x)),
        &Expr::Annotate(ref r, e, ref ty) =>
            match check(ctx, e, ty) {
                Ok(()) => Ok(ty.clone()),
                Err(err) => Err(TypeError::bad_annotation(r.clone(), e, ty.clone(), err)),
            },
        &Expr::App(ref r, e1, e2) =>
            match synthesize(ctx, e1)? {
                Type::Function(ty_a, ty_b) => {
                    match check(ctx, e2, &ty_a) {
                        Ok(()) => Ok(*ty_b),
                        Err(arg_err) => Err(TypeError::bad_argument(r.clone(), *ty_a, e1, e2, arg_err)),
                    }
                },
                ty =>
                    Err(TypeError::non_function_application(r.clone(), e1, ty)),
            },
        &Expr::Force(ref r, e1) =>
            match synthesize(ctx, e1)? {
                Type::Later(ty) => Ok(*ty),
                // TODO: add error here
                ty => Err(TypeError::forcing_non_thunk(r.clone(), e1, ty)),
            },
        &Expr::LetIn(ref r, x, e1, e2) =>
            match synthesize(ctx, e1) {
                Ok(ty_x) => {
                    let mut new_ctx = ctx.clone();
                    new_ctx.insert(x, ty_x);
                    synthesize(&new_ctx, e2)
                },
                Err(err) =>
                    Err(TypeError::let_failure(r.clone(), x, e1, err)),
            },
        &Expr::UnPair(ref r, x1, x2, e0, e) =>
            match synthesize(ctx, e0)? {
                Type::Product(ty1, ty2) => {
                    let mut new_ctx = ctx.clone();
                    new_ctx.insert(x1, *ty1);
                    new_ctx.insert(x2, *ty2);
                    synthesize(&new_ctx, e)
                },
                ty =>
                    Err(TypeError::unpairing_non_product(r.clone(), e0, ty)),
            }
        &Expr::Case(ref r, e0, x1, e1, x2, e2) =>
            match synthesize(ctx, e0)? {
                Type::Sum(ty1, ty2) => {
                    let mut ctx1 = ctx.clone();
                    ctx1.insert(x1, *ty1);
                    let ty_out1 = synthesize(&ctx1, e1)?;

                    let mut ctx2 = ctx.clone();
                    ctx2.insert(x2, *ty2);
                    let ty_out2 = synthesize(&ctx2, e2)?;

                    meet(ty_out1, ty_out2)
                },
                ty =>
                    Err(TypeError::casing_non_sum(r.clone(), e0, ty)),
            },
        _ =>
            Err(TypeError::unsupported(expr)),
    }
}

fn meet<'a, R>(ty1: Type, ty2: Type) -> Result<Type, TypeError<'a, R>> {
    if ty1 == ty2 {
        Ok(ty1)
    } else {
        Err(TypeError::could_not_unify(ty1, ty2))
    }
}

#[cfg(test)]
mod test {
    use crate::expr::Value;
    use super::*;
    
    fn s(i: usize) -> Symbol { string_interner::Symbol::try_from_usize(i).unwrap() }

    #[test]
    fn try_out() {
        let ev = Expr::Val((), Value::Unit);
        let e = Expr::Annotate((), &ev, Type::Unit);

        assert_eq!(synthesize(&Ctx::new(), &e).unwrap(), Type::Unit);
    }

    #[test]
    fn test_fn() {
        let e_x = Expr::Var((), s(0));
        let e = Expr::Lam((), s(0), &e_x);

        assert!(check(&Ctx::new(), &e, &Type::Function(Box::new(Type::Unit), Box::new(Type::Unit))).is_ok());
        assert!(check(&Ctx::new(), &e, &Type::Function(Box::new(Type::Index), Box::new(Type::Unit))).is_err());
    }
}
