use std::collections::HashMap;
use std::fmt;

use string_interner::{StringInterner, DefaultSymbol, DefaultStringInterner};

use typed_arena::Arena;

type Symbol = DefaultSymbol;

#[derive(Debug, Clone)]
enum Value<'a> {
    Sample(f32),
    Index(usize),
    Gen(Env<'a>, Box<Value<'a>>, &'a Expr<'a, ()>),
    Closure(Env<'a>, Symbol, &'a Expr<'a, ()>),
    Suspend(Env<'a>, &'a Expr<'a, ()>),
}

// TODO: use a better type here... eventually we should resolve symbols and just use de bruijn offsets or similar...
type Env<'a> = HashMap<Symbol, Value<'a>>;
// type Env<'a> = imbl::HashMap<Symbol, Value<'a>>;

#[derive(Debug, Clone)]
enum Expr<'a, R> {
    Var(R, Symbol),
    Val(R, Value<'a>),
    Lam(R, Symbol, &'a Expr<'a, R>),
    App(R, &'a Expr<'a, R>, &'a Expr<'a, R>),
    Force(R, &'a Expr<'a, R>),
    Lob(R, Symbol, &'a Expr<'a, R>),
    Gen(R, &'a Expr<'a, R>, &'a Expr<'a, R>),
}

impl<'a, R> Expr<'a, R> {
    fn map_ext<'b, U>(&self, arena: &'b Arena<Expr<'b, U>>, f: &dyn Fn(&R) -> U) -> Expr<'b, U> where 'a: 'b {
        match *self {
            Expr::Var(ref r, s) => Expr::Var(f(r), s),
            Expr::Val(ref r, ref v) => Expr::Val(f(r), v.clone()),
            Expr::Lam(ref r, s, ref e) => Expr::Lam(f(r), s, arena.alloc(e.map_ext(arena, f))),
            Expr::App(ref r, ref e1, ref e2) => Expr::App(f(r), arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
            Expr::Force(ref r, ref e) => Expr::Force(f(r), arena.alloc(e.map_ext(arena, f))),
            Expr::Lob(ref r, s, ref e) => Expr::Lob(f(r), s, arena.alloc(e.map_ext(arena, f))),
            Expr::Gen(ref r, ref e1, ref e2) => Expr::Gen(f(r), arena.alloc(e1.map_ext(arena, f)), arena.alloc(e2.map_ext(arena, f))),
        }
    }
}
        

fn interp<'a>(env: &Env<'a>, expr: &'a Expr<'a, ()>) -> Result<Value<'a>, &'static str> {
    match *expr {
        Expr::Var(_, ref x) => Ok(env.get(x).ok_or("where's the var")?.clone()),
        Expr::Val(_, ref v) => Ok(v.clone()),
        Expr::Lam(_, ref x, ref e) => Ok(Value::Closure(env.clone(), x.clone(), *e)),
        Expr::App(_, ref e1, ref e2) => {
            let v1 = interp(env, &*e1)?;
            let Value::Closure(env1, x, ebody) = v1 else {
                return Err("don't call something that's not a closure!!");
            };
            let v2 = interp(env, &*e2)?;
            let mut new_env = env1.clone();
            new_env.insert(x, v2);
            // TODO: i don't think rust has TCE and this can't be
            // TCE'd anyway bc new_env must be deallocated.
            interp(&new_env, &*ebody)
        },
        Expr::Force(_, ref e) => {
            let v = interp(env, &*e)?;
            let Value::Suspend(env1, ebody) = v else {
                return Err("don't force something that's not a suspension!!");
            };
            interp(&env1, &*ebody)
        },
        Expr::Lob(_, s, ref e) => {
            let susp = Value::Suspend(env.clone(), expr);
            let mut new_env = env.clone();
            new_env.insert(s, susp);
            interp(&new_env, e)
        },
        Expr::Gen(_, ref eh, ref et) => {
            let vh = interp(env, eh)?;
            Ok(Value::Gen(env.clone(), Box::new(vh), et))
        },
    }
}

macro_rules! make_node_enum {
    ($enum_name:ident { $($rust_name:ident : $ts_name:ident),* } with matcher $matcher_name:ident) => {
        #[derive(PartialEq, Eq, Debug, Copy, Clone)]
        enum $enum_name {
            $( $rust_name ),*
        }

        struct $matcher_name {
            node_id_table: Vec<Option<$enum_name>>,
        }

        impl $matcher_name {
            fn new(lang: &tree_sitter::Language) -> $matcher_name {
                let mut table = [None].repeat(lang.node_kind_count());
                $( table[lang.id_for_node_kind(stringify!($ts_name), true) as usize] = Some($enum_name::$rust_name); )*
                $matcher_name {
                    node_id_table: table
                }
            }

            fn lookup(&self, id: u16) -> Option<$enum_name> {
                self.node_id_table.get(id as usize).copied().flatten()
            }
        }
    };
}

// why isn't this information in the generated bindings...?
make_node_enum!(ConcreteNode {
    SourceFile: source_file,
    Expression: expression,
    WrapExpression: wrap_expression,
    Identifier: identifier,
    Literal: literal,
    Sample: sample,
    ApplicationExpression: application_expression,
    LambdaExpression: lambda_expression,
    LobExpression: lob_expression,
    ForceExpression: force_expression,
    GenExpression: gen_expression
} with matcher ConcreteNodeMatcher);

struct AbstractionContext<'a, 'b> {
    original_text: &'a str,
    node_matcher: ConcreteNodeMatcher,
    interner: &'a mut DefaultStringInterner,
    arena: &'b Arena<Expr<'b, tree_sitter::Range>>,
}

impl<'a, 'b> AbstractionContext<'a, 'b> {
    fn node_text<'c>(&self, node: tree_sitter::Node<'c>) -> &'a str {
        // utf8_text must return Ok because it is fetching from a &str, which must be utf8
        node.utf8_text(self.original_text.as_bytes()).unwrap()
    }

    fn parse_concrete<'c>(&mut self, node: tree_sitter::Node<'c>) -> Result<Expr<'b, tree_sitter::Range>, String> {
        // TODO: use a TreeCursor instead
        match self.node_matcher.lookup(node.kind_id()) {
            Some(ConcreteNode::SourceFile) =>
                self.parse_concrete(node.child(0).unwrap()),
            Some(ConcreteNode::Expression) =>
                self.parse_concrete(node.child(0).unwrap()),
            Some(ConcreteNode::WrapExpression) =>
                // the literals are included in the children indices
                self.parse_concrete(node.child(1).unwrap()),
            Some(ConcreteNode::Identifier) => {
                let interned_ident = self.interner.get_or_intern(self.node_text(node));
                Ok(Expr::Var(node.range(), interned_ident))
            },
            Some(ConcreteNode::Literal) => {
                let int_lit = self.node_text(node).parse().map_err(|_| "int lit out of bounds".to_string())?;
                Ok(Expr::Val(node.range(), Value::Index(int_lit)))
            },
            Some(ConcreteNode::Sample) => {
                let sample_text = self.node_text(node);
                let sample = sample_text.parse().map_err(|_| format!("couldn't parse sample {:?}", sample_text))?;
                Ok(Expr::Val(node.range(), Value::Sample(sample)))
            },
            Some(ConcreteNode::ApplicationExpression) => {
                let e1 = self.parse_concrete(node.child(0).unwrap())?;
                let e2 = self.parse_concrete(node.child(1).unwrap())?;
                Ok(Expr::App(node.range(), self.arena.alloc(e1), self.arena.alloc(e2)))
            },
            Some(ConcreteNode::LambdaExpression) => {
                let x = self.interner.get_or_intern(self.node_text(node.child(1).unwrap()));
                let e = self.parse_concrete(node.child(3).unwrap())?;
                Ok(Expr::Lam(node.range(), x, self.arena.alloc(e)))
            },
            Some(ConcreteNode::LobExpression) => {
                let x = self.interner.get_or_intern(self.node_text(node.child(1).unwrap()));
                let e = self.parse_concrete(node.child(3).unwrap())?;
                Ok(Expr::Lob(node.range(), x, self.arena.alloc(e)))
            },
            Some(ConcreteNode::ForceExpression) => {
                let e = self.parse_concrete(node.child(1).unwrap())?;
                Ok(Expr::Force(node.range(), self.arena.alloc(e)))
            },
            Some(ConcreteNode::GenExpression) => {
                let e1 = self.parse_concrete(node.child(0).unwrap())?;
                let e2 = self.parse_concrete(node.child(2).unwrap())?;
                Ok(Expr::Gen(node.range(), self.arena.alloc(e1), self.arena.alloc(e2)))
            },
            None => {
                eprintln!("{:?}", node);
                Err("what".to_string())
            },
        }
    }
}

struct PrettyExpr<'a, 'b, R> {
    interner: &'a DefaultStringInterner,
    expr: &'a Expr<'b, R>,
}

impl<'a, 'b, R> PrettyExpr<'a, 'b, R> {
    fn for_expr(&self, other_expr: &'a Expr<'b, R>) -> PrettyExpr<'a, 'b, R> {
        PrettyExpr { interner: self.interner, expr: other_expr }
    }
}

impl<'a, 'b, R> fmt::Display for PrettyExpr<'a, 'b, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.expr {
            Expr::Var(_, x) =>
                write!(f, "Var({})", self.interner.resolve(x).unwrap()),
            Expr::Val(_, ref v) =>
                write!(f, "{:?}", v),
            Expr::App(_, ref e1, ref e2) =>
                write!(f, "App({}, {})", self.for_expr(e1), self.for_expr(e2))
            ,
            Expr::Lam(_, x, ref e) => {
                let x_str = self.interner.resolve(x).unwrap();
                write!(f, "Lam({}, {})", x_str, self.for_expr(e))
            },
            Expr::Force(_, ref e) =>
                write!(f, "Force({})", self.for_expr(e)),
            Expr::Lob(_, x, ref e) => {
                let x_str = self.interner.resolve(x).unwrap();
                write!(f, "Lob({}, {})", x_str, self.for_expr(e))
            },
            Expr::Gen(_, ref eh, ref et) =>
                write!(f, "Gen({}, {})", self.for_expr(eh), self.for_expr(et)),
        }
    }
}

fn get_samples<'a>(mut expr: &'a Expr<'a, ()>, out: &mut [f32]) -> Result<(), String> {
    let mut env = Env::new();
    for (i, s_out) in out.iter_mut().enumerate() {
        match interp(&env, expr) {
            Ok(Value::Gen(new_env, head, next_expr)) => {
                if let Value::Sample(s) = *head {
                    env = new_env;
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

fn main() {
    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_lambdalisten::language();
    let node_id_matcher = ConcreteNodeMatcher::new(&lang);
    parser.set_language(lang).expect("Error loading lambda listen grammar");

    let source_code = r"(&s. \x. x :: 2.4 :: !s 0.0) 5.3";
    let tree = parser.parse(source_code, None).unwrap();
    let root_node = tree.root_node();

    println!("\n{:?}", node_id_matcher.node_id_table);

    let annot_arena = Arena::new();
    let mut interner = StringInterner::new();
    let mut abs_context = AbstractionContext {
        original_text: source_code,
        node_matcher: node_id_matcher,
        interner: &mut interner,
        arena: &annot_arena
    };

    println!("\n{:?}", tree);
    println!("\n{:?}", root_node.kind_id());

    let expr = abs_context.parse_concrete(root_node).unwrap();
    drop(abs_context);
    println!("\n{}", PrettyExpr { interner: &interner, expr: &expr });

    let arena = Arena::new();
    let expr_unannotated = expr.map_ext(&arena, &(|_| ()));

    // println!("{:?}", interp(&Env::new(), &expr_unannotated));

    let mut samples = [0f32; 8];
    println!("\n{:?}", get_samples(&expr_unannotated, &mut samples[..]));
    println!("{samples:?}");
}
