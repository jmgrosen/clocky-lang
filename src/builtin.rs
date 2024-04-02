use std::collections::HashMap;

use string_interner::DefaultStringInterner;

use crate::expr::{Symbol, Value};

type BuiltinFn = for<'a> fn(&[Value<'a>]) -> Result<Value<'a>, &'static str>;

#[derive(Clone, Copy, Debug)]
pub struct Builtin {
    pub n_args: usize,
    pub body: BuiltinFn,
}

impl Builtin {
    const fn new(n_args: usize, body: BuiltinFn) -> Builtin {
        Builtin { n_args, body }
    }
}

macro_rules! builtins {
    ( $($name:ident [ $nargs:literal ] { $pat:pat => $e:expr }),* ) => {
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
                (stringify!($name), Builtin::new($nargs, builtins::$name)),
            )*
        ];
    }
}

builtins!(
    pi[0] { &[] => Ok(Value::Sample(std::f32::consts::PI)) },
    sin[1] { &[Value::Sample(s)] => Ok(Value::Sample(s.sin())) },
    cos[1] { &[Value::Sample(s)] => Ok(Value::Sample(s.cos())) },
    add[2] { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 + s2)) },
    sub[2] { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 - s2)) },
    mul[2] { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 * s2)) },
    div[2] { &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 / s2)) }
);

pub type BuiltinsMap = HashMap<Symbol, Builtin>;

pub fn make_builtins(interner: &mut DefaultStringInterner) -> BuiltinsMap {
    BUILTINS.iter().map(|&(name, builtin)| (interner.get_or_intern_static(name), builtin)).collect()
}
