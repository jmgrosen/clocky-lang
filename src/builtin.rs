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

fn builtin_pi<'a>(args: &[Value<'a>]) -> Result<Value<'a>, &'static str> {
    match args {
        &[] => Ok(Value::Sample(std::f32::consts::PI)),
        _ => Err("pi called on something different than zero values"),
    }
}

fn builtin_sin<'a>(args: &[Value<'a>]) -> Result<Value<'a>, &'static str> {
    match args {
        &[Value::Sample(s)] => Ok(Value::Sample(s.sin())),
        _ => Err("sin called on something different than one sample"),
    }
}

fn builtin_add<'a>(args: &[Value<'a>]) -> Result<Value<'a>, &'static str> {
    match args {
        &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 + s2)),
        _ => Err("add called on something different than two samples"),
    }
}

fn builtin_div<'a>(args: &[Value<'a>]) -> Result<Value<'a>, &'static str> {
    match args {
        &[Value::Sample(s1), Value::Sample(s2)] => Ok(Value::Sample(s1 / s2)),
        _ => Err("div called on something different than two samples"),
    }
}

const BUILTINS: &[(&'static str, Builtin)] = &[
    ("pi", Builtin::new(0, builtin_pi)),
    ("sin", Builtin::new(1, builtin_sin)),
    ("add", Builtin::new(2, builtin_add)),
    ("div", Builtin::new(2, builtin_div)),
];

pub type BuiltinsMap = HashMap<Symbol, Builtin>;

pub fn make_builtins(interner: &mut DefaultStringInterner) -> BuiltinsMap {
    BUILTINS.iter().map(|&(name, builtin)| (interner.get_or_intern_static(name), builtin)).collect()
}
