use std::collections::HashMap;

use crate::expr::{Binop, Expr, Value, Env, Symbol};
use crate::builtin::BuiltinsMap;

pub struct InterpretationContext<'a, 'b> {
    pub builtins: &'b BuiltinsMap,
    pub defs: &'b HashMap<Symbol, &'a Expr<'a, ()>>,
    pub env: Env<'a>,
}

impl<'a, 'b> InterpretationContext<'a, 'b> {
    fn with_env(&self, new_env: Env<'a>) -> InterpretationContext<'a, 'b> {
        InterpretationContext { env: new_env, ..*self }
    }
}

fn interp_binop<'a>(op: Binop, v1: Value<'a>, v2: Value<'a>) -> Result<Value<'a>, &'static str> {
    match (v1, v2) {
        (Value::Sample(x1), Value::Sample(x2)) =>
            Ok(Value::Sample(match op {
                Binop::FMul => x1 * x2,
                Binop::FDiv => x1 / x2,
                Binop::FAdd => x1 + x2,
                Binop::FSub => x1 - x2,
                _ => return Err("cannot do some weird binop on two samples"),
            })),
        (Value::Index(i1), Value::Index(i2)) =>
            Ok(Value::Index(match op {
                Binop::IMul => i1 * i2,
                Binop::IDiv => i1 / i2,
                Binop::IAdd => i1 + i2,
                Binop::ISub => i1 - i2,
                Binop::Shl => i1 << i2,
                Binop::Shr => i1 >> i2,
                Binop::And => i1 & i2,
                Binop::Xor => i1 ^ i2,
                Binop::Or => i1 | i2,
                _ => return Err("cannot do some weird binop on two indices"),
            })),
        (_, _) =>
            Err("bad binop combo"),
    }
}

pub fn interp<'a, 'b>(ctx: &InterpretationContext<'a, 'b>, expr: &'a Expr<'a, ()>) -> Result<Value<'a>, &'static str> {
    match *expr {
        Expr::Var(_, ref x) => {
            if let Some(v) = ctx.env.get(x) {
                Ok(v.clone())
            } else if let Some(e) = ctx.defs.get(x) {
                interp(&ctx.with_env(Env::new()), e)
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
        Expr::Adv(_, ref e) => {
            let v = interp(ctx, &*e)?;
            let Value::Suspend(env1, ebody) = v else {
                return Err("don't force something that's not a suspension!!");
            };
            interp(&ctx.with_env(env1), &*ebody)
        },
        Expr::Lob(_, _, s, ref e) => {
            // TODO: is this really the right semantics?
            let boxed = Value::BoxDelay(ctx.env.clone(), expr);
            let mut new_env = ctx.env.clone();
            new_env.insert(s, boxed);
            interp(&ctx.with_env(new_env), e)
        },
        Expr::Gen(_, ref eh, ref et) => {
            let vh = interp(ctx, eh)?;
            // now this is expected to result in some sort of fix expression or smth
            let vt = interp(ctx, et)?;
            Ok(Value::Gen(Box::new(vh), Box::new(vt)))
        },
        Expr::LetIn(_, x, _, e1, e2) => {
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
                Value::Gen(v_hd, v_tl) =>
                    Ok(Value::Pair(v_hd, v_tl)),
                _ =>
                    Err("tried to ungen a non-gen"),
            },
        Expr::Delay(_, e) =>
            Ok(Value::Suspend(ctx.env.clone(), e)),
        Expr::Box(_, e) =>
            Ok(Value::Box(ctx.env.clone(), e)),
        Expr::Unbox(_, e) =>
            match interp(ctx, e)? {
                Value::Box(new_env, e_body) =>
                    interp(&ctx.with_env(new_env), e_body),
                Value::BoxDelay(new_env, e_body) =>
                    Ok(Value::Suspend(new_env, e_body)),
                _ =>
                    Err("tried to unbox a non-box"),
            },
        Expr::ClockApp(_, e, _) =>
            interp(ctx, e),
        Expr::TypeApp(_, e, _) =>
            interp(ctx, e),
        Expr::Binop(_, op, e1, e2) =>
            interp_binop(op, interp(ctx, e1)?, interp(ctx, e2)?),
    }
}

pub fn get_samples<'a>(builtins: &'a BuiltinsMap, defs: &'a HashMap<Symbol, &'a Expr<'a, ()>>, mut expr: &'a Expr<'a, ()>, out: &mut [f32]) -> Result<(), String> {
    let mut ctx = InterpretationContext { builtins, defs, env: Env::new() };
    for (i, s_out) in out.iter_mut().enumerate() {
        match interp(&ctx, expr) {
            Ok(Value::Gen(head, tail)) => {
                match (*head, *tail) {
                    (Value::Sample(s), Value::Suspend(new_env, next_expr)) => {
                        ctx.env = new_env;
                        *s_out = s;
                        expr = next_expr;
                    },
                    (h, t) =>
                        return Err(format!("on index {i}, evaluation succeeded with a Gen but got head {h:?} and tail {t:?}")),
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
