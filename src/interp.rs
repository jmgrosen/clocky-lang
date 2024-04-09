use crate::expr::{Expr, Value, Env};
use crate::builtin::BuiltinsMap;

pub struct InterpretationContext<'a, 'b> {
    pub builtins: &'b BuiltinsMap,
    pub env: Env<'a>,
}

impl<'a, 'b> InterpretationContext<'a, 'b> {
    fn with_env(&self, new_env: Env<'a>) -> InterpretationContext<'a, 'b> {
        InterpretationContext { builtins: self.builtins, env: new_env }
    }
}

pub fn interp<'a, 'b>(ctx: &InterpretationContext<'a, 'b>, expr: &'a Expr<'a, ()>) -> Result<Value<'a>, &'static str> {
    match *expr {
        Expr::Var(_, ref x) => {
            if let Some(v) = ctx.env.get(x) {
                Ok(v.clone())
            } else if let Some(&builtin) = ctx.builtins.get(x) {
                if builtin.n_args == 0 {
                    (builtin.body)(&[])
                } else {
                    Ok(Value::BuiltinPartial(builtin, [].into()))
                }
            } else {
                Err("couldn't find the var in locals or builtins")
            }
        },
        Expr::Val(_, ref v) =>
            Ok(v.clone()),
        Expr::Annotate(_, e, _) =>
            interp(ctx, e),
        Expr::Lam(_, ref x, ref e) =>
            Ok(Value::Closure(ctx.env.clone(), x.clone(), *e)),
        Expr::App(_, ref e1, ref e2) => {
            let v1 = interp(ctx, &*e1)?;
            let v2 = interp(ctx, &*e2)?;
            match v1 {
                Value::Closure(env1, x, ebody) => {
                    let mut new_env = env1.clone();
                    new_env.insert(x, v2);
                    // TODO: i don't think rust has TCE and this can't be
                    // TCE'd anyway bc new_env must be deallocated.
                    interp(&ctx.with_env(new_env), &*ebody)
                },
                Value::BuiltinPartial(builtin, args_so_far) => {
                    let mut new_args = Vec::with_capacity(args_so_far.len() + 1);
                    new_args.extend(args_so_far.iter().cloned());
                    new_args.push(v2);
                    if builtin.n_args == new_args.len() {
                        (builtin.body)(&new_args[..])
                    } else {
                        Ok(Value::BuiltinPartial(builtin, new_args.into()))
                    }
                },
                _ =>
                    Err("don't call something that's not a closure!!"),
            }
        },
        Expr::Force(_, ref e) => {
            let v = interp(ctx, &*e)?;
            let Value::Suspend(env1, ebody) = v else {
                return Err("don't force something that's not a suspension!!");
            };
            interp(&ctx.with_env(env1), &*ebody)
        },
        Expr::Lob(_, s, ref e) => {
            let susp = Value::Suspend(ctx.env.clone(), expr);
            let mut new_env = ctx.env.clone();
            new_env.insert(s, susp);
            interp(&ctx.with_env(new_env), e)
        },
        Expr::Gen(_, ref eh, ref et) => {
            let vh = interp(ctx, eh)?;
            Ok(Value::Gen(ctx.env.clone(), Box::new(vh), et))
        },
        Expr::LetIn(_, x, e1, e2) => {
            let v = interp(ctx, e1)?;
            let mut new_env = ctx.env.clone();
            new_env.insert(x, v);
            interp(&ctx.with_env(new_env), e2)
        },
        Expr::Pair(_, e1, e2) => {
            let v1 = interp(ctx, e1)?;
            let v2 = interp(ctx, e2)?;
            Ok(Value::Pair(Box::new(v1), Box::new(v2)))
        },
        Expr::UnPair(_, x1, x2, e0, e) => {
            let Value::Pair(v1, v2) = interp(ctx, e0)? else {
                return Err("tried to unpair a non-pair!");
            };
            let mut new_env = ctx.env.clone();
            new_env.insert(x1, *v1);
            new_env.insert(x2, *v2);
            interp(&ctx.with_env(new_env), e)
        },
        Expr::InL(_, e) =>
            Ok(Value::InL(Box::new(interp(ctx, e)?))),
        Expr::InR(_, e) =>
            Ok(Value::InR(Box::new(interp(ctx, e)?))),
        Expr::Case(_, e0, x1, e1, x2, e2) =>
            match interp(ctx, e0)? {
                Value::InL(v) => {
                    let mut new_env = ctx.env.clone();
                    new_env.insert(x1, *v);
                    interp(&ctx.with_env(new_env), e1)
                },
                Value::InR(v) => {
                    let mut new_env = ctx.env.clone();
                    new_env.insert(x2, *v);
                    interp(&ctx.with_env(new_env), e2)
                },
                _ =>
                    Err("tried to case on a non-sum"),
            },
        Expr::Array(_, ref es) => {
            let mut vs = Vec::with_capacity(es.len());
            for e in es.iter() {
                vs.push(interp(ctx, e)?);
            }
            Ok(Value::Array(vs.into()))
        },
        Expr::UnGen(_, e) =>
            match interp(ctx, e)? {
                Value::Gen(new_env, v_hd, e_tl) =>
                    Ok(Value::Pair(v_hd, Box::new(Value::Suspend(new_env, e_tl)))),
                _ =>
                    Err("tried to ungen a non-gen"),
            },
    }
}

pub fn get_samples<'a>(builtins: &'a BuiltinsMap, mut expr: &'a Expr<'a, ()>, out: &mut [f32]) -> Result<(), String> {
    let mut ctx = InterpretationContext { builtins, env: Env::new() };
    for (i, s_out) in out.iter_mut().enumerate() {
        match interp(&ctx, expr) {
            Ok(Value::Gen(new_env, head, next_expr)) => {
                if let Value::Sample(s) = *head {
                    ctx.env = new_env;
                    *s_out = s;
                    expr = next_expr;
                } else {
                    return Err(format!("on index {i}, evaluation succeeded with a Gen but got head {head:?}"));
                }
            },
            Ok(v) => {
                return Err(format!("on index {i}, evaluation succeeded but got {v:?}"));
            },
            Err(e) => {
                return Err(format!("on index {i}, evaluation failed with error {e:?}"));
            },
        }
    }
    Ok(())
}
