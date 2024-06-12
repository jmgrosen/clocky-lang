use std::collections::HashMap;

use string_interner::DefaultStringInterner;

use crate::expr::Symbol;
use crate::ir1::{DebruijnIndex, Op};
use crate::ir2;
use crate::typing::{Clock, Kind, Type};

type BuiltinTypeFn = fn(&mut DefaultStringInterner) -> Type;

#[derive(Clone, Copy, Debug)]
pub struct Builtin<Ty> {
    pub n_args: usize,
    // there's gotta be a better way to do this
    pub type_: Ty,
    pub ir2_expr: &'static ir2::Expr<'static>,
}

impl<Ty> Builtin<Ty> {
    const fn new(n_args: usize, type_: Ty, ir2_expr: &'static ir2::Expr<'static>) -> Builtin<Ty> {
        Builtin { n_args, type_, ir2_expr }
    }
}

macro_rules! builtins {
    ( $($name:ident [ $nargs:literal ] { $interner:pat => $type_expr:expr } [ $ir2e:expr ] ),* $(,)? ) => {
        mod builtin_types {
            use super::*;

            $(
                pub fn $name($interner: &mut DefaultStringInterner) -> Type {
                    $type_expr
                }
            )*
        }
        const BUILTINS: &[(&'static str, Builtin<BuiltinTypeFn>)] = &[
            $(
                (stringify!($name), Builtin::new($nargs, builtin_types::$name, $ir2e)),
            )*
        ];
    }
}

fn g(interner: &mut DefaultStringInterner, name: &'static str) -> Symbol {
    interner.get_or_intern_static(name)
}

/*
const fn alloc_f32(e: &'static ir2::Expr<'static>) -> &'static ir2::Expr<'static> {
    &ir2::Expr::Op(Op::AllocF32, &[e])
}
*/

macro_rules! alloc_f32 {
    ( $e:expr ) => {
        &ir2::Expr::Op(Op::AllocF32, &[$e])
    }
}

macro_rules! alloc_i32 {
    ( $e:expr ) => {
        &ir2::Expr::Op(Op::AllocI32, &[$e])
    }
}

macro_rules! deref_f32 {
    ( $e:expr ) => {
        &ir2::Expr::Op(Op::DerefF32, &[$e])
    }
}

macro_rules! deref_i32 {
    ( $e:expr ) => {
        &ir2::Expr::Op(Op::DerefI32, &[$e])
    }
}

builtins!(
    pi[0]
      { _ => Type::Sample }
      [ alloc_f32!(&ir2::Expr::Op(Op::Pi, &[])) ],
    sin[1]
      { _ => Type::Function(Type::Sample.into(), Type::Sample.into()) }
      [ alloc_f32!(&ir2::Expr::Op(Op::Sin, &[deref_f32!(&ir2::Expr::Var(DebruijnIndex(0)))])) ],
    cos[1]
      { _ => Type::Function(Type::Sample.into(), Type::Sample.into()) }
      [ alloc_f32!(&ir2::Expr::Op(Op::Cos, &[deref_f32!(&ir2::Expr::Var(DebruijnIndex(0)))])) ],
    reinterpi[1]
      { _ => Type::Function(Type::Index.into(), Type::Sample.into()) }
      [ alloc_f32!(&ir2::Expr::Op(Op::ReinterpI2F, &[deref_i32!(&ir2::Expr::Var(DebruijnIndex(0)))])) ],
    reinterpf[1]
      { _ => Type::Function(Type::Sample.into(), Type::Index.into()) }
      [ alloc_i32!(&ir2::Expr::Op(Op::ReinterpF2I, &[deref_f32!(&ir2::Expr::Var(DebruijnIndex(0)))])) ],
    cast[1]
      { _ => Type::Function(Type::Index.into(), Type::Sample.into()) }
      [ alloc_f32!(&ir2::Expr::Op(Op::CastI2F, &[deref_i32!(&ir2::Expr::Var(DebruijnIndex(0)))])) ],
    since_tick[1]
      { i => Type::Forall(g(i, "c"), Kind::Clock, Type::Stream(Clock::from_var(g(i, "c")), Type::Sample.into()).into()) }
      [ &ir2::Expr::Op(Op::SinceLastTickStream, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    wait[1]
      { i => Type::Forall(g(i, "c"), Kind::Clock, Type::Later(Clock::from_var(g(i, "c")), Type::Unit.into()).into()) }
      [ &ir2::Expr::Op(Op::Wait, &[&ir2::Expr::Var(DebruijnIndex(0))]) ],
    sched[3]
      { i =>
          Type::Forall(g(i, "a"), Kind::Type, Box::new(
              Type::Forall(g(i, "c"), Kind::Clock, Box::new(
                  Type::Forall(g(i, "d"), Kind::Clock, Box::new(
                      Type::Function(Box::new(
                          Type::Later(Clock::from_var(g(i, "c")), Box::new(Type::TypeVar(g(i, "a"))))
                      ), Box::new(
                          Type::Later(Clock::from_var(g(i, "d")), Box::new(
                              Type::Sum(Box::new(
                                  Type::Unit
                              ), Box::new(
                                  Type::TypeVar(g(i, "a"))
                              ))
                          ))
                      ))
                  ))
              ))
          ))
      }
      [ &ir2::Expr::Op(Op::Schedule, &[&ir2::Expr::Var(DebruijnIndex(2)), &ir2::Expr::Var(DebruijnIndex(1)), &ir2::Expr::Var(DebruijnIndex(0))]) ],
);

pub type BuiltinsMap = HashMap<Symbol, Builtin<Type>>;

pub fn make_builtins(interner: &mut DefaultStringInterner) -> BuiltinsMap {
    BUILTINS.iter().map(|&(name, builtin)| {
        let builtin_constructed = Builtin {
            n_args: builtin.n_args,
            type_: (builtin.type_)(interner),
            ir2_expr: builtin.ir2_expr,
        };
        (interner.get_or_intern_static(name), builtin_constructed)
    }).collect()
}

// TODO: need to synchronize the order of these with the
// runtime. maybe we could fetch these from the runtime somehow? hmmmm
const BUILTIN_CLOCKS: &[&'static str] = &[
    "audio",
];

pub fn make_builtin_clocks(interner: &mut DefaultStringInterner) -> Vec<Symbol> {
    BUILTIN_CLOCKS.iter().map(|&name| interner.get_or_intern_static(name)).collect()
}
