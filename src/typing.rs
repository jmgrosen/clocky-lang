use std::collections::HashMap;

use crate::expr::{Symbol, Expr, Value};

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

type Ctx = HashMap<Symbol, Type>;

#[derive(Debug)]
pub enum TypeError<'a, R> {
    MismatchingTypes { expr: &'a Expr<'a, R>, synth: Type, expected: Type },
    VariableNotFound { range: R, var: Symbol },
    BadArgument { range: R, fun: &'a Expr<'a, R>, arg: &'a Expr<'a, R>, arg_err: Box<TypeError<'a, R>> },
    BadApplication { range: R, purported_fun: &'a Expr<'a, R>, actual_type: Type },
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

    fn bad_argument(range: R, fun: &'a Expr<'a, R>, arg: &'a Expr<'a, R>, arg_err: TypeError<'a, R>) -> TypeError<'a, R> {
        TypeError::BadArgument { range, fun, arg, arg_err: Box::new(arg_err) }
    }

    fn bad_application(range: R, purported_fun: &'a Expr<'a, R>, actual_type: Type) -> TypeError<'a, R> {
        TypeError::BadApplication { range, purported_fun, actual_type }
    }

    fn unsupported(expr: &'a Expr<'a, R>) -> TypeError<'a, R> {
        TypeError::Unsupported { expr }
    }

    fn bad_annotation(range: R, expr: &'a Expr<'a, R>, purported_type: Type, err: TypeError<'a, R>) -> TypeError<'a, R> {
        TypeError::BadAnnotation { range, expr, purported_type, err: Box::new(err) }
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
                    match check(ctx, e1, &ty_a) {
                        Ok(()) => Ok(*ty_b),
                        Err(arg_err) => Err(TypeError::bad_argument(r.clone(), e1, e2, arg_err)),
                    }
                },
                ty =>
                    Err(TypeError::bad_application(r.clone(), e1, ty)),
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
