use core::fmt;
use std::collections::HashMap;

use string_interner::{StringInterner, DefaultStringInterner};

use crate::expr::{Symbol, Expr, Value, PrettyExpr};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Type {
    Unit,
    Sample,
    Index,
    Stream(Box<Type>),
    Function(Box<Type>, Box<Type>),
    Product(Box<Type>, Box<Type>),
    Later(Box<Type>),
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
        &Expr::Val(ref r, ref v) =>
            match *v {
                Value::Unit => Ok(Type::Unit),
                Value::Sample(_) => Ok(Type::Sample),
                Value::Index(_) => Ok(Type::Index),
                Value::Gen(_, _, _) |
                Value::Closure(_, _, _) |
                Value::Suspend(_, _) |
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
        _ =>
            Err(TypeError::unsupported(expr)),
    }
}

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
