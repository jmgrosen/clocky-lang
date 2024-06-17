use imbl::HashMap;
use num::rational::Ratio;
use ordered_float::OrderedFloat;
use egglog::{ast::{Literal, Symbol}, Term, TermDag, match_term_app};

use crate::{ir1::{Con, DebruijnIndex, Expr, Global, Op, Value}, util::ArenaPlus};

// this file uses a lot from eggcc

pub struct ToEgglogConverter {
    pub termdag: TermDag,
}

impl ToEgglogConverter {
    pub fn new() -> ToEgglogConverter {
        ToEgglogConverter {
            termdag: TermDag::default(),
        }
    }

    fn lit_int(&mut self, i: i64) -> Term {
        self.termdag.lit(Literal::Int(i))
    }

    fn lit_float(&mut self, x: f64) -> Term {
        self.termdag.lit(Literal::F64(OrderedFloat(x)))
    }

    fn app(&mut self, f: Symbol, args: Vec<Term>) -> Term {
        self.termdag.app(f, args)
    }

    #[allow(unused)]
    fn var(&mut self, v: Symbol) -> Term {
        self.termdag.var(v)
    }

    pub fn expr_to_term<'a>(&mut self, e: &'a Expr<'a>) -> Term {
        match *e {
            Expr::Var(DebruijnIndex(i)) => {
                let i = self.lit_int(i as i64);
                self.app("Var".into(), vec![i])
            }
            Expr::Val(Value::Unit) => {
                let unit = self.app("VUnit".into(), vec![]);
                self.app("Val".into(), vec![unit])
            },
            Expr::Val(Value::Index(i)) => {
                let i = self.lit_int(i as i64);
                let idx = self.app("Index".into(), vec![i]);
                self.app("Val".into(), vec![idx])
            },
            Expr::Val(Value::Sample(x)) => {
                let x = self.lit_float(x as f64); // TODO: hmmmm
                let samp = self.app("Sample".into(), vec![x]);
                self.app("Val".into(), vec![samp])
            },
            Expr::Glob(Global(g)) => {
                let g = self.lit_int(g as i64);
                self.app("Glob".into(), vec![g])
            },
            Expr::Lam(None, 1, body) => {
                let body = self.expr_to_term(body);
                self.app("Lam".into(), vec![body])
            },
            Expr::Lam(_, _, _) =>
                panic!("can only handle arity-1 lams for now!"),
            Expr::App(f, &[arg]) => {
                let f = self.expr_to_term(f);
                let arg = self.expr_to_term(arg);
                self.app("App".into(), vec![f, arg])
            },
            Expr::App(_, _) =>
                panic!("can only handle arity-1 apps for now!"),
            Expr::Unbox(e) => {
                let e = self.expr_to_term(e);
                self.app("Unbox".into(), vec![e])
            },
            Expr::Box(None, body) => {
                let body = self.expr_to_term(body);
                self.app("Box".into(), vec![body])
            },
            Expr::Box(_, _) =>
                panic!("can only handle untransformed boxes (for now?)"),
            Expr::Lob(None, body) => {
                let body = self.expr_to_term(body);
                self.app("Lob".into(), vec![body])
            },
            Expr::Lob(_, _) =>
                panic!("can only handle untransformed lobs (for now?)"),
            Expr::LetIn(e1, e2) => {
                let e1 = self.expr_to_term(e1);
                let e2 = self.expr_to_term(e2);
                self.app("Let".into(), vec![e1, e2])
            },
            Expr::If(e0, e1, e2) => {
                let e0 = self.expr_to_term(e0);
                let e1 = self.expr_to_term(e1);
                let e2 = self.expr_to_term(e2);
                self.app("If".into(), vec![e0, e1, e2])
            },
            Expr::Con(con, args) => {
                let con = self.con_to_term(con);
                let args = self.exprs_to_list(args);
                self.app("Con".into(), vec![con, args])
            },
            Expr::Op(op, args) => {
                let op = self.op_to_term(op);
                let args = self.exprs_to_list(args);
                self.app("Op".into(), vec![op, args])
            },
            Expr::Delay(None, body) => {
                let body = self.expr_to_term(body);
                self.app("Delay".into(), vec![body])
            },
            Expr::Delay(_, _) =>
                panic!("can only handle untransformed delays (for now?)"),
            Expr::Adv(e) => {
                let e = self.expr_to_term(e);
                self.app("Adv".into(), vec![e])
            },
        }
    }

    fn exprs_to_list<'a>(&mut self, exprs: &[&'a Expr<'a>]) -> Term {
        let mut l = self.app("ELNil".into(), vec![]);
        for e in exprs {
            let t = self.expr_to_term(e);
            l = self.app("ELCons".into(), vec![t, l]);
        }
        l
    }

    fn con_to_term(&mut self, con: Con) -> Term {
        match con {
            Con::Stream => self.app("Stream".into(), vec![]),
            Con::Array => self.app("Array".into(), vec![]),
            Con::Pair => self.app("Pair".into(), vec![]),
            Con::InL => self.app("InL".into(), vec![]),
            Con::InR => self.app("InR".into(), vec![]),
            Con::ClockEx => self.app("ClockEx".into(), vec![]),
        }
    }

    fn op_to_term(&mut self, op: Op) -> Term {
        match op {
            Op::Const(v) => {
                let args = vec![self.value_to_term(v)];
                self.app("Const".into(), args)
            },
            Op::FAdd => self.app("FAdd".into(), vec![]),
            Op::FSub => self.app("FSub".into(), vec![]),
            Op::FMul => self.app("FMul".into(), vec![]),
            Op::FDiv => self.app("FDiv".into(), vec![]),
            Op::FGt => self.app("FGt".into(), vec![]),
            Op::FGe => self.app("FGe".into(), vec![]),
            Op::FLt => self.app("FLt".into(), vec![]),
            Op::FLe => self.app("FLe".into(), vec![]),
            Op::FEq => self.app("FEq".into(), vec![]),
            Op::FNe => self.app("FNe".into(), vec![]),
            Op::Sin => self.app("Sin".into(), vec![]),
            Op::Cos => self.app("Cos".into(), vec![]),
            Op::Pi => self.app("Pi".into(), vec![]),
            Op::IAdd => self.app("IAdd".into(), vec![]),
            Op::ISub => self.app("ISub".into(), vec![]),
            Op::IMul => self.app("IMul".into(), vec![]),
            Op::IDiv => self.app("IDiv".into(), vec![]),
            Op::Shl => self.app("Shl".into(), vec![]),
            Op::Shr => self.app("Shr".into(), vec![]),
            Op::And => self.app("And".into(), vec![]),
            Op::Xor => self.app("Xor".into(), vec![]),
            Op::Or => self.app("Or".into(), vec![]),
            Op::IGt => self.app("IGt".into(), vec![]),
            Op::IGe => self.app("IGe".into(), vec![]),
            Op::ILt => self.app("ILt".into(), vec![]),
            Op::ILe => self.app("ILe".into(), vec![]),
            Op::IEq => self.app("IEq".into(), vec![]),
            Op::INe => self.app("INe".into(), vec![]),
            Op::ReinterpF2I => self.app("ReinterpF2I".into(), vec![]),
            Op::ReinterpI2F => self.app("ReinterpI2F".into(), vec![]),
            Op::CastI2F => self.app("CastI2F".into(), vec![]),
            Op::Proj(i) => {
                let args = vec![self.lit_int(i as i64)];
                self.app("Proj".into(), args)
            },
            Op::UnGen => self.app("UnGen".into(), vec![]),
            Op::AllocAndFill => self.app("AllocAndFill".into(), vec![]),
            Op::AllocF32 => self.app("AllocF32".into(), vec![]),
            Op::AllocI32 => self.app("AllocI32".into(), vec![]),
            Op::BuildClosure(_) => todo!(),
            Op::LoadGlobal(_) => todo!(),
            Op::DerefF32 => self.app("DerefF32".into(), vec![]),
            Op::DerefI32 => self.app("DerefI32".into(), vec![]),
            Op::ApplyCoeff(c) => {
                let n = self.lit_int(*c.numer() as i64);
                let d = self.lit_int(*c.denom() as i64);
                self.app("ApplyCoeff".into(), vec![n, d])
            },
            Op::SinceLastTickStream => self.app("SinceLastTickStream".into(), vec![]),
            Op::Advance => self.app("Advance".into(), vec![]),
            Op::Wait => self.app("Wait".into(), vec![]),
            Op::Schedule => self.app("Schedule".into(), vec![]),
            Op::MakeClock(f) => {
                let args = vec![self.lit_float(f as f64)];
                self.app("MakeClock".into(), args)
            },
            Op::GetClock(i) => {
                let args = vec![self.lit_int(i as i64)];
                self.app("GetClock".into(), args)
            },
        }
    }

    fn value_to_term(&mut self, v: Value) -> Term {
        match v {
            Value::Unit => self.app("VUnit".into(), vec![]),
            Value::Sample(x) => {
                let args = vec![self.lit_float(x as f64)];
                self.app("Sample".into(), args)
            },
            Value::Index(i) => {
                let args = vec![self.lit_int(i as i64)];
                self.app("Index".into(), args)
            },
        }
    }
}

pub struct FromEgglogConverter<'a> {
    pub termdag: TermDag,
    pub arena: &'a ArenaPlus<'a, Expr<'a>>,
    pub expr_cache: HashMap<Term, &'a Expr<'a>>,
    pub slice_cache: HashMap<Term, &'a [&'a Expr<'a>]>,
}

impl<'a> FromEgglogConverter<'a> {
    pub fn term_to_expr(&mut self, t: Term) -> &'a Expr<'a> {
        if let Some(&e) = self.expr_cache.get(&t) {
            return e;
        }
        let expr = match_term_app!(t.clone(); {
            ("Var", &[l]) => self.arena.alloc(Expr::Var(DebruijnIndex(self.lit_term_to_int(self.termdag.get(l))))),
            ("Val", &[v]) => self.arena.alloc(Expr::Val(self.val_term_to_value(self.termdag.get(v)))),
            ("Glob", &[i]) => self.arena.alloc(Expr::Glob(Global(self.lit_term_to_int(self.termdag.get(i))))),
            ("Lam", &[body_term]) => {
                let body = self.term_to_expr(self.termdag.get(body_term));
                self.arena.alloc(Expr::Lam(None, 1, body))
            },
            ("App", &[func_term, arg_term]) => {
                let func = self.term_to_expr(self.termdag.get(func_term));
                let arg = self.term_to_expr(self.termdag.get(arg_term));
                // TODO: slice cache?
                self.arena.alloc(Expr::App(func, self.arena.alloc_slice([arg])))
            },
            ("Box", &[body_term]) => {
                let body = self.term_to_expr(self.termdag.get(body_term));
                self.arena.alloc(Expr::Box(None, body))
            },
            ("Unbox", &[body_term]) => {
                let body = self.term_to_expr(self.termdag.get(body_term));
                self.arena.alloc(Expr::Unbox(body))
            },
            ("Lob", &[body_term]) => {
                let body = self.term_to_expr(self.termdag.get(body_term));
                self.arena.alloc(Expr::Lob(None, body))
            },
            ("Let", &[bindee_term, body_term]) => {
                let bindee = self.term_to_expr(self.termdag.get(bindee_term));
                let body = self.term_to_expr(self.termdag.get(body_term));
                self.arena.alloc(Expr::LetIn(bindee, body))
            },
            ("If", &[scrutinee_term, true_term, false_term]) => {
                let scrutinee = self.term_to_expr(self.termdag.get(scrutinee_term));
                let true_ = self.term_to_expr(self.termdag.get(true_term));
                let false_ = self.term_to_expr(self.termdag.get(false_term));
                self.arena.alloc(Expr::If(scrutinee, true_, false_))
            },
            ("Delay", &[body_term]) => {
                let body = self.term_to_expr(self.termdag.get(body_term));
                self.arena.alloc(Expr::Delay(None, body))
            },
            ("Adv", &[body_term]) => {
                let body = self.term_to_expr(self.termdag.get(body_term));
                self.arena.alloc(Expr::Adv(body))
            },
            ("Op", &[op_term, args_term]) => {
                let op = self.term_to_op(self.termdag.get(op_term));
                let args = self.term_to_expr_slice(self.termdag.get(args_term));
                self.arena.alloc(Expr::Op(op, args))
            },
            ("Con", &[con_term, args_term]) => {
                let con = self.term_to_con(self.termdag.get(con_term));
                let args = self.term_to_expr_slice(self.termdag.get(args_term));
                self.arena.alloc(Expr::Con(con, args))
            },
            _ => panic!("Invalid expr"),
        });
        self.expr_cache.insert(t, expr);
        expr
    }

    fn term_to_expr_slice(&mut self, t: Term) -> &'a [&'a Expr<'a>] {
        if let Some(&slice) = self.slice_cache.get(&t) {
            return slice;
        }
        let mut exprs = Vec::new();
        let mut cur = t.clone();
        loop {
            match_term_app!(cur; {
                ("ELCons", &[head_term, tail_term]) => {
                    let head = self.term_to_expr(self.termdag.get(head_term));
                    exprs.push(head);
                    cur = self.termdag.get(tail_term);
                },
                ("ELNil", &[]) => {
                    break;
                },
                _ => panic!("Invalid expr list"),
            });
        }
        let slice = self.arena.alloc_slice(exprs);
        self.slice_cache.insert(t, slice);
        slice
    }

    fn lit_term_to_int(&self, t: Term) -> u32 {
        match t {
            Term::Lit(Literal::Int(i)) => i as u32,
            _ => panic!("invalid int lit"),
        }
    }

    fn lit_term_to_float(&self, t: Term) -> f32 {
        match t {
            Term::Lit(Literal::F64(x)) => *x as f32,
            _ => panic!("invalid float lit"),
        }
    }

    fn val_term_to_value(&self, t: Term) -> Value {
        match_term_app!(t; {
            ("VUnit", &[]) => Value::Unit,
            ("Index", &[t]) => Value::Index(self.lit_term_to_int(self.termdag.get(t)) as usize),
            ("Sample", &[t]) => Value::Sample(self.lit_term_to_float(self.termdag.get(t))),
            _ => panic!("weird value app"),
        })
    }

    fn term_to_op(&self, t: Term) -> Op {
        match_term_app!(t; {
            ("FAdd", &[]) => Op::FAdd,
            ("FSub", &[]) => Op::FSub,
            ("FMul", &[]) => Op::FMul,
            ("FDiv", &[]) => Op::FDiv,
            ("FGt", &[]) => Op::FGt,
            ("FGe", &[]) => Op::FGe,
            ("FLt", &[]) => Op::FLt,
            ("FLe", &[]) => Op::FLe,
            ("FEq", &[]) => Op::FEq,
            ("FNe", &[]) => Op::FNe,
            ("Sin", &[]) => Op::Sin,
            ("Cos", &[]) => Op::Cos,
            ("Pi", &[]) => Op::Pi,
            ("IAdd", &[]) => Op::IAdd,
            ("ISub", &[]) => Op::ISub,
            ("IMul", &[]) => Op::IMul,
            ("IDiv", &[]) => Op::IDiv,
            ("Shl", &[]) => Op::Shl,
            ("Shr", &[]) => Op::Shr,
            ("And", &[]) => Op::And,
            ("Xor", &[]) => Op::Xor,
            ("Or", &[]) => Op::Or,
            ("IGt", &[]) => Op::IGt,
            ("IGe", &[]) => Op::IGe,
            ("ILt", &[]) => Op::ILt,
            ("ILe", &[]) => Op::ILe,
            ("IEq", &[]) => Op::IEq,
            ("INe", &[]) => Op::INe,
            ("ReinterpF2I", &[]) => Op::ReinterpF2I,
            ("ReinterpI2F", &[]) => Op::ReinterpI2F,
            ("CastI2F", &[]) => Op::CastI2F,
            ("UnGen", &[]) => Op::UnGen,
            ("AllocAndFill", &[]) => Op::AllocAndFill,
            ("AllocF32", &[]) => Op::AllocF32,
            ("AllocI32", &[]) => Op::AllocI32,
            ("DerefF32", &[]) => Op::DerefF32,
            ("DerefI32", &[]) => Op::DerefI32,
            ("SinceLastTickStream", &[]) => Op::SinceLastTickStream,
            ("Advance", &[]) => Op::Advance,
            ("Wait", &[]) => Op::Wait,
            ("Schedule", &[]) => Op::Schedule,
            ("Proj", &[i]) => Op::Proj(self.lit_term_to_int(self.termdag.get(i))),
            ("ApplyCoeff", &[n, d]) => Op::ApplyCoeff(Ratio::new(self.lit_term_to_int(self.termdag.get(n)),
                                                                 self.lit_term_to_int(self.termdag.get(d)))),
            ("MakeClock", &[f]) => Op::MakeClock(self.lit_term_to_float(self.termdag.get(f))),
            ("GetClock", &[i]) => Op::GetClock(self.lit_term_to_int(self.termdag.get(i))),
            (op, args) => panic!("unknown op {} or bad args {:?}", op, args),
        })
    }

    fn term_to_con(&self, t: Term) -> Con {
        match_term_app!(t; {
            ("Stream", &[]) => Con::Stream,
            ("Array", &[]) => Con::Array,
            ("Pair", &[]) => Con::Pair,
            ("InL", &[]) => Con::InL,
            ("InR", &[]) => Con::InR,
            ("ClockEx", &[]) => Con::ClockEx,
            (con, args) => panic!("unknown con {} and args {:?}", con, args),
        })
    }
}
