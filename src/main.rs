use std::collections::HashMap;
use std::rc::Rc;
use std::fmt;

use string_interner::{StringInterner, DefaultSymbol, DefaultStringInterner};

type Symbol = DefaultSymbol;

#[derive(Debug, Clone)]
enum Value {
    Sample(f32),
    Index(usize),
    Gen(Env, Rc<Expr<()>>, Rc<Expr<()>>),
    Closure(Env, Symbol, Rc<Expr<()>>),
}

type Env = HashMap<Symbol, Value>;

#[derive(Debug, Clone)]
enum Expr<R> {
    Var(R, Symbol),
    Val(R, Value),
    Lam(R, Symbol, Rc<Expr<R>>),
    App(R, Rc<Expr<R>>, Rc<Expr<R>>),
}

impl<R> Expr<R> {
    fn map_ext<U>(&self, f: &dyn Fn(&R) -> U) -> Expr<U> {
        match *self {
            Expr::Var(ref r, ref s) => Expr::Var(f(r), s.clone()),
            Expr::Val(ref r, ref v) => Expr::Val(f(r), v.clone()),
            Expr::Lam(ref r, ref s, ref e) => Expr::Lam(f(r), s.clone(), Rc::new(e.map_ext(f))),
            Expr::App(ref r, ref e1, ref e2) => Expr::App(f(r), Rc::new(e1.map_ext(f)), Rc::new(e2.map_ext(f))),
        }
    }
}
        

fn interp(env: &Env, expr: &Expr<()>) -> Result<Value, &'static str> {
    match *expr {
        Expr::Var(_, ref x) => Ok(env.get(x).ok_or("where's the var")?.clone()),
        Expr::Val(_, ref v) => Ok(v.clone()),
        Expr::Lam(_, ref x, ref e) => Ok(Value::Closure(env.clone(), x.clone(), e.clone())),
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
        }
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
    ApplicationExpression: application_expression,
    LambdaExpression: lambda_expression
} with matcher ConcreteNodeMatcher);

struct AbstractionContext<'a> {
    original_text: &'a str,
    node_matcher: ConcreteNodeMatcher,
    interner: DefaultStringInterner,
}

impl<'a> AbstractionContext<'a> {
    fn node_text<'b>(&self, node: tree_sitter::Node<'b>) -> &'a str {
        // utf8_text must return Ok because it is fetching from a &str, which must be utf8
        node.utf8_text(self.original_text.as_bytes()).unwrap()
    }

    fn parse_concrete<'b>(&mut self, node: tree_sitter::Node<'b>) -> Result<Expr<tree_sitter::Range>, String> {
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
            Some(ConcreteNode::ApplicationExpression) => {
                let e1 = self.parse_concrete(node.child(0).unwrap())?;
                let e2 = self.parse_concrete(node.child(1).unwrap())?;
                Ok(Expr::App(node.range(), e1.into(), e2.into()))
            },
            Some(ConcreteNode::LambdaExpression) => {
                let x = self.interner.get_or_intern(self.node_text(node.child(1).unwrap()));
                let e = self.parse_concrete(node.child(3).unwrap())?;
                Ok(Expr::Lam(node.range(), x, e.into()))
            },
            None => {
                eprintln!("{:?}", node);
                Err("what".to_string())
            },
        }
    }
}

struct PrettyExpr<'a, R> {
    interner: &'a DefaultStringInterner,
    expr: &'a Expr<R>,
}

impl<'a, R> fmt::Display for PrettyExpr<'a, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.expr {
            Expr::Var(_, ref x) => write!(f, "Var({})", self.interner.resolve(*x).unwrap()),
            Expr::Val(_, ref v) => write!(f, "{:?}", v),
            Expr::App(_, ref e1, ref e2) => {
                write!(f, "App({}, ", PrettyExpr { interner: self.interner, expr: e1 })?;
                write!(f, "{})", PrettyExpr { interner: self.interner, expr: e2 })
            },
            Expr::Lam(_, ref x, ref e) => {
                let x_str = self.interner.resolve(*x).unwrap();
                write!(f, "Lam({}, {})", x_str, PrettyExpr { interner: self.interner, expr: e })
            }
        }
    }
}

fn main() {
    let mut interner = StringInterner::default();
    let x = interner.get_or_intern("x");
    let e = Expr::App((), Rc::new(Expr::Lam((), x, Rc::new(Expr::Var((), x)))), Rc::new(Expr::Val((), Value::Sample(2.0))));
    println!("{:?}", e);
    println!("{:?}", interp(&HashMap::new(), &e));


    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_lambdalisten::language();
    let node_id_matcher = ConcreteNodeMatcher::new(&lang);
    parser.set_language(lang).expect("Error loading lambda listen grammar");

    let source_code = r"(\x. x 4321) (\y. y)";
    let tree = parser.parse(source_code, None).unwrap();
    let root_node = tree.root_node();

    println!("\n{:?}", node_id_matcher.node_id_table);

    let mut abs_context = AbstractionContext {
        original_text: source_code,
        node_matcher: node_id_matcher,
        interner
    };

    println!("\n{:?}", tree);
    println!("\n{:?}", root_node.kind_id());

    let expr = match abs_context.parse_concrete(root_node) {
        Ok(e) => {
            println!("\n{}", PrettyExpr { interner: &abs_context.interner, expr: &e });
            e
        },
        Err(e) => {
            println!("\nerror: {}", e);
            return;
        },
    };

    let expr_unannotated = expr.map_ext(&(|_| ()));
    println!("{:?}", interp(&HashMap::new(), &expr_unannotated));
}
