use std::collections::HashMap;

use string_interner::DefaultStringInterner;

use crate::expr::Symbol;
use crate::ir1::{DebruijnIndex, Op};
use crate::ir2;

#[derive(Clone, Copy, Debug)]
pub struct Builtin {
    pub n_args: usize,
    pub ir2_expr: &'static ir2::Expr<'static>,
}

impl Builtin {
    const fn new(n_args: usize, ir2_expr: &'static ir2::Expr<'static>) -> Builtin {
        Builtin { n_args, ir2_expr }
    }
}

macro_rules! builtins {
    ( $($name:ident [ $nargs:literal ] [ $ir2e:expr ] ),* $(,)? ) => {
        const BUILTINS: &[(&'static str, Builtin)] = &[
            $(
                (stringify!($name), Builtin::new($nargs, $ir2e)),
            )*
        ];
    }
}

builtins!(
    pi[0]
      [ &ir2::Expr::Op(Op::Pi, &[]) ],
    sin[1]
      [ &ir2::Expr::Op(Op::Sin, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    cos[1]
      [ &ir2::Expr::Op(Op::Cos, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    add[2]
      [ &ir2::Expr::Op(Op::FAdd, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    sub[2]
      // TODO: make sure this order is correct?
      [ &ir2::Expr::Op(Op::FSub, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    mul[2]
      [ &ir2::Expr::Op(Op::FMul, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    div[2]
      [ &ir2::Expr::Op(Op::FDiv, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Var(DebruijnIndex(1))]) ],
    addone[1]
      [ &ir2::Expr::Op(Op::FAdd, &[&ir2::Expr::Var(DebruijnIndex(0)), &ir2::Expr::Op(Op::Const(crate::ir1::Value::Sample(1.0)), &[])]) ],
    reinterpi[1]
      [ &ir2::Expr::Op(Op::ReinterpI2F, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    reinterpf[1]
      [ &ir2::Expr::Op(Op::ReinterpF2I, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    cast[1]
      [ &ir2::Expr::Op(Op::CastI2F, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    since_tick[1]
      [ &ir2::Expr::Op(Op::SinceLastTickStream, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    wait[1]
      [ &ir2::Expr::Op(Op::Wait, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    sched[3]
      [ &ir2::Expr::Op(Op::Schedule, &[&ir2::Expr::Var(DebruijnIndex(2)), &ir2::Expr::Var(DebruijnIndex(1)), &ir2::Expr::Var(DebruijnIndex(0))]) ],
);

pub type BuiltinsMap = HashMap<Symbol, Builtin>;

pub fn make_builtins(interner: &mut DefaultStringInterner) -> BuiltinsMap {
    BUILTINS.iter().map(|&(name, builtin)| (interner.get_or_intern_static(name), builtin)).collect()
}
