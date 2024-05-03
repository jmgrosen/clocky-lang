use std::collections::HashMap;

use string_interner::DefaultStringInterner;

use crate::expr::{Symbol, Value};
use crate::ir1::{DebruijnIndex, Op};
use crate::ir2;

type BuiltinFn = for<'a> fn(&[Value<'a>]) -> Result<Value<'a>, &'static str>;

#[derive(Clone, Copy, Debug)]
pub struct Builtin {
    pub n_args: usize,
    pub body: BuiltinFn,
    pub ir2_expr: &'static ir2::Expr<'static>,
}

impl Builtin {
    const fn new(n_args: usize, body: BuiltinFn, ir2_expr: &'static ir2::Expr<'static>) -> Builtin {
        Builtin { n_args, body, ir2_expr }
    }
}

macro_rules! builtins {
    ( $($name:ident [ $nargs:literal ] { $pat:pat => $e:expr } [ $ir2e:expr ] ),* $(,)? ) => {
        mod builtins {
            use super::*;

            $(
                pub fn $name<'a>(args: &[Value<'a>]) -> Result<Value<'a>, &'static str> {
                    match args {
                        $pat => $e,
                        _ => Err(concat!("args passed to ", stringify!($name), " do not match the pattern ", stringify!($pat))),
                    }
                }
            )*
        }
        const BUILTINS: &[(&'static str, Builtin)] = &[
            $(
                (stringify!($name), Builtin::new($nargs, builtins::$name, $ir2e)),
            )*
        ];
    }
}

builtins!(
    pi[0]
      { &[] => Ok(Value::Sample(std::f32::consts::PI)) }
      [ &ir2::Expr::Op(Op::Pi, &[]) ],
    sin[1]
      { &[Value::Sample(s)] => Ok(Value::Sample(s.sin())) }
      [ &ir2::Expr::Op(Op::Sin, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    cos[1]
      { &[Value::Sample(s)] => Ok(Value::Sample(s.cos())) }
      [ &ir2::Expr::Op(Op::Cos, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    add[2]
      { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 + s2)) }
      [ &ir2::Expr::Op(Op::FAdd, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    sub[2]
      { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 - s2)) }
      // TODO: make sure this order is correct?
      [ &ir2::Expr::Op(Op::FSub, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    mul[2]
      { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 * s2)) }
      [ &ir2::Expr::Op(Op::FMul, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    div[2]
      { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 / s2)) }
      [ &ir2::Expr::Op(Op::FDiv, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    addone[1]
      { &[Value::Sample(s)] => Ok(Value::Sample(s + 1.0)) }
      [ &ir2::Expr::Op(Op::FAdd, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Op(Op::Const(crate::ir1::Value::Sample(1.0)), &[])]) ],
    reinterpi[1]
      { &[Value::Index(i)] => Ok(Value::Sample(f32::from_bits(i as u32))) }
      [ &ir2::Expr::Op(Op::ReinterpI2F, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    reinterpf[1]
      { &[Value::Sample(x)] => Ok(Value::Index(x.to_bits() as usize)) }
      [ &ir2::Expr::Op(Op::ReinterpF2I, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    cast[1]
      { &[Value::Index(i)] => Ok(Value::Sample(i as f32)) }
      [ &ir2::Expr::Op(Op::CastI2F, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    // sub[2] { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 - s2)) },
    // mul[2] { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 * s2)) },
    // div[2] { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 / s2)) }
);

pub type BuiltinsMap = HashMap<Symbol, Builtin>;

pub fn make_builtins(interner: &mut DefaultStringInterner) -> BuiltinsMap {
    BUILTINS.iter().map(|&(name, builtin)| (interner.get_or_intern_static(name), builtin)).collect()
}
