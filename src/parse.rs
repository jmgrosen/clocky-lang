use string_interner::DefaultStringInterner;
use typed_arena::Arena;
use num::rational::Ratio;

use crate::{expr::{Binop, Expr, SourceFile, Symbol, TopLevelDef, TopLevelDefKind, Value}, typing::{Type, ArraySize, Clock, Kind}};

macro_rules! make_node_enum {
    ($enum_name:ident { $($rust_name:ident : $ts_name:ident),* } with matcher $matcher_name:ident) => {
        #[derive(PartialEq, Eq, Debug, Copy, Clone)]
        enum $enum_name {
            $( $rust_name ),*
        }

        pub struct $matcher_name {
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
    TopLevelDef: top_level_def,
    TopLevelLet: top_level_let,
    Expression: expression,
    WrapExpression: wrap_expression,
    Identifier: identifier,
    Literal: literal,
    Sample: sample,
    ApplicationExpression: application_expression,
    LambdaExpression: lambda_expression,
    LobExpression: lob_expression,
    ForceExpression: force_expression,
    GenExpression: gen_expression,
    LetExpression: let_expression,
    AnnotateExpression: annotate_expression,
    PairExpression: pair_expression,
    UnPairExpression: unpair_expression,
    InLExpression: inl_expression,
    InRExpression: inr_expression,
    CaseExpression: case_expression,
    ArrayExpression: array_expression,
    ArrayInner: array_inner,
    UnGenExpression: ungen_expression,
    UnitExpression: unit_expression,
    DelayExpression: delay_expression,
    BoxExpression: box_expression,
    UnboxExpression: unbox_expression,
    ClockAppExpression: clockapp_expression,
    TypeAppExpression: typeapp_expression,
    BinopExpression: binop_expression,
    Type: type,
    WrapType: wrap_type,
    BaseType: base_type,
    FunctionType: function_type,
    StreamType: stream_type,
    ProductType: product_type,
    SumType: sum_type,
    ArrayType: array_type,
    LaterType: later_type,
    BoxType: box_type,
    ForallType: forall_type,
    VarType: var_type,
    Kind: kind
} with matcher ConcreteNodeMatcher);

pub struct Parser<'a, 'b> {
    parser: tree_sitter::Parser,
    node_matcher: ConcreteNodeMatcher,
    pub interner: &'a mut DefaultStringInterner,
    pub arena: &'b Arena<Expr<'b, tree_sitter::Range>>,
}

impl<'a, 'b> Parser<'a, 'b> {
    pub fn new(interner: &'a mut DefaultStringInterner, arena: &'b Arena<Expr<'b, tree_sitter::Range>>) -> Parser<'a, 'b> {
        let mut parser = tree_sitter::Parser::new();
        let lang = tree_sitter_clocky::language();
        let node_matcher = ConcreteNodeMatcher::new(&lang);
        parser.set_language(lang).expect("Error loading clocky grammar");

        Parser {
            parser,
            node_matcher,
            interner,
            arena,
        }
    }

    pub fn parse_file(&mut self, text: &str) -> Result<SourceFile<'b, tree_sitter::Range>, FullParseError> {
        // this unwrap should be safe because we make sure to set the language and don't set a timeout or cancellation flag
        let tree = self.parser.parse(text, None).unwrap();
        let root_node = tree.root_node();
        AbstractionContext { parser: self, original_text: text }
            .parse_file(root_node)
            .map_err(|error| FullParseError { tree, error })
    }
}

#[derive(Debug)]
pub struct FullParseError {
    pub tree: tree_sitter::Tree,
    pub error: ParseError,
}

#[derive(Debug)]
pub enum ParseError {
    BadLiteral(tree_sitter::Range),
    ExpectedExpression(tree_sitter::Range),
    ExpectedType(tree_sitter::Range),
    UnknownNodeType(tree_sitter::Range, String),
    BadCoefficient(tree_sitter::Range),
    UhhhhhhWhat(tree_sitter::Range, String),
}

struct AbstractionContext<'a, 'b, 'c> {
    parser: &'c mut Parser<'a, 'b>,
    original_text: &'c str,
}

impl<'a, 'b, 'c> AbstractionContext<'a, 'b, 'c> {
    fn node_text<'d>(&self, node: tree_sitter::Node<'d>) -> &'c str {
        // utf8_text must return Ok because it is fetching from a &str, which must be utf8
        node.utf8_text(self.original_text.as_bytes()).unwrap()
    }

    fn alloc(&self, expr: Expr<'b, tree_sitter::Range>) -> &'b Expr<'b, tree_sitter::Range> {
        self.parser.arena.alloc(expr)
    }

    fn identifier<'d>(&mut self, node: tree_sitter::Node<'d>) -> Symbol {
        self.parser.interner.get_or_intern(self.node_text(node))
    }

    fn parse_file<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<SourceFile<'b, tree_sitter::Range>, ParseError> {
        let Some(ConcreteNode::SourceFile) = self.parser.node_matcher.lookup(node.kind_id()) else {
            return Err(ParseError::UhhhhhhWhat(node.range(), "you didn't pass me a file".to_string()));
        };

        let mut defs = Vec::new();
        let mut cur = node.walk();
        cur.goto_first_child();
        loop {
            defs.push(self.parse_top_level_let(cur.node())?);
            if !cur.goto_next_sibling() {
                break;
            }
        }

        Ok(SourceFile { defs })
    }

    fn parse_top_level_let<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<TopLevelDef<'b, tree_sitter::Range>, ParseError> {
        let kind = match self.parser.node_matcher.lookup(node.kind_id()) {
            Some(ConcreteNode::TopLevelDef) => TopLevelDefKind::Def,
            Some(ConcreteNode::TopLevelLet) => TopLevelDefKind::Let,
            _ => return Err(ParseError::UhhhhhhWhat(node.range(), "expected a top-level let here".to_string()))
        };

        let body = self.parse_expr(node.child(5).unwrap())?;
        Ok(TopLevelDef {
            kind,
            name: self.identifier(node.child(1).unwrap()),
            type_: self.parse_type(node.child(3).unwrap())?,
            body: self.alloc(body),
            range: node.range(),
        })
    }

    fn parse_expr<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<Expr<'b, tree_sitter::Range>, ParseError> {
        // TODO: use a TreeCursor instead
        match self.parser.node_matcher.lookup(node.kind_id()) {
            Some(ConcreteNode::Expression) =>
                self.parse_expr(node.child(0).unwrap()),
            Some(ConcreteNode::WrapExpression) =>
                // the literals are included in the children indices
                self.parse_expr(node.child(1).unwrap()),
            Some(ConcreteNode::Identifier) => {
                let interned_ident = self.parser.interner.get_or_intern(self.node_text(node));
                Ok(Expr::Var(node.range(), interned_ident))
            },
            Some(ConcreteNode::Literal) => {
                let text = self.node_text(node);
                let int_lit = (if text.starts_with("0x") {
                    usize::from_str_radix(&text[2..], 16)
                } else {
                    text.parse()
                }).map_err(|_| ParseError::BadLiteral(node.range()))?;
                Ok(Expr::Val(node.range(), Value::Index(int_lit)))
            },
            Some(ConcreteNode::Sample) => {
                let sample_text = self.node_text(node);
                let sample = sample_text.parse().map_err(|_| ParseError::BadLiteral(node.range()))?;
                Ok(Expr::Val(node.range(), Value::Sample(sample)))
            },
            Some(ConcreteNode::ApplicationExpression) => {
                let e1 = self.parse_expr(node.child(0).unwrap())?;
                let e2 = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::App(node.range(), self.parser.arena.alloc(e1), self.parser.arena.alloc(e2)))
            },
            Some(ConcreteNode::LambdaExpression) => {
                let x = self.parser.interner.get_or_intern(self.node_text(node.child(1).unwrap()));
                let e = self.parse_expr(node.child(3).unwrap())?;
                Ok(Expr::Lam(node.range(), x, self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::LobExpression) => {
                let clock = self.parse_clock(node.child(3).unwrap())?;
                let x = self.parser.interner.get_or_intern(self.node_text(node.child(5).unwrap()));
                let e = self.parse_expr(node.child(7).unwrap())?;
                Ok(Expr::Lob(node.range(), clock, x, self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::ForceExpression) => {
                let e = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::Adv(node.range(), self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::GenExpression) => {
                let e1 = self.parse_expr(node.child(0).unwrap())?;
                let e2 = self.parse_expr(node.child(2).unwrap())?;
                Ok(Expr::Gen(node.range(), self.parser.arena.alloc(e1), self.parser.arena.alloc(e2)))
            },
            Some(ConcreteNode::LetExpression) => {
                let x = self.parser.interner.get_or_intern(self.node_text(node.child(1).unwrap()));
                let (ty, e1_idx, e2_idx) = match node.child_count() {
                    6 => (None, 3, 5),
                    8 => {
                        let ty = self.parse_type(node.child(3).unwrap())?;
                        (Some(ty), 5, 7)
                    },
                    n => {
                        let msg = format!("how does a let expression have {} children??", n);
                        return Err(ParseError::UhhhhhhWhat(node.range(), msg));
                    },
                };
                let e1 = self.parse_expr(node.child(e1_idx).unwrap())?;
                let e2 = self.parse_expr(node.child(e2_idx).unwrap())?;
                Ok(Expr::LetIn(node.range(), x, ty, self.parser.arena.alloc(e1), self.parser.arena.alloc(e2)))
            },
            Some(ConcreteNode::AnnotateExpression) => {
                let e = self.parse_expr(node.child(0).unwrap())?;
                let ty = self.parse_type(node.child(2).unwrap())?;
                Ok(Expr::Annotate(node.range(), self.parser.arena.alloc(e), ty))
            },
            Some(ConcreteNode::PairExpression) => {
                let e1 = self.parse_expr(node.child(1).unwrap())?;
                let e2 = self.parse_expr(node.child(3).unwrap())?;
                Ok(Expr::Pair(node.range(), self.alloc(e1), self.alloc(e2)))
            },
            Some(ConcreteNode::UnPairExpression) => {
                let x1 = self.identifier(node.child(2).unwrap());
                let x2 = self.identifier(node.child(4).unwrap());
                let e0 = self.parse_expr(node.child(7).unwrap())?;
                let e = self.parse_expr(node.child(9).unwrap())?;
                Ok(Expr::UnPair(node.range(), x1, x2, self.alloc(e0), self.alloc(e)))
            },
            Some(ConcreteNode::InLExpression) => {
                let e = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::InL(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::InRExpression) => {
                let e = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::InR(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::CaseExpression) => {
                let e0 = self.parse_expr(node.child(1).unwrap())?;
                let x1 = self.identifier(node.child(4).unwrap());
                let e1 = self.parse_expr(node.child(6).unwrap())?;
                let x2 = self.identifier(node.child(9).unwrap());
                let e2 = self.parse_expr(node.child(11).unwrap())?;
                Ok(Expr::Case(node.range(), self.alloc(e0), x1, self.alloc(e1), x2, self.alloc(e2)))
            },
            Some(ConcreteNode::ArrayExpression) => {
                Ok(Expr::Array(node.range(),
                    if node.child_count() == 2 {
                        [].into()
                    } else {
                        let array_inner = node.child(1).unwrap();
                        let mut es = Vec::with_capacity((array_inner.child_count() + 1) / 2);
                        let mut cur = array_inner.walk();
                        cur.goto_first_child();
                        loop {
                            let e = self.parse_expr(cur.node())?;
                            es.push(self.alloc(e));
                            cur.goto_next_sibling();
                            if !cur.goto_next_sibling() {
                                break;
                            }
                        }
                        es.into()
                    }))
            },
            Some(ConcreteNode::UnGenExpression) => {
                let e = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::UnGen(node.range(), self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::UnitExpression) =>
                Ok(Expr::Val(node.range(), Value::Unit)),
            Some(ConcreteNode::DelayExpression) => {
                let e = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::Delay(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::BoxExpression) => {
                let e = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::Box(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::UnboxExpression) => {
                let e = self.parse_expr(node.child(1).unwrap())?;
                Ok(Expr::Unbox(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::ClockAppExpression) => {
                let e = self.parse_expr(node.child(0).unwrap())?;
                let clock = self.parse_clock(node.child(3).unwrap())?;
                Ok(Expr::ClockApp(node.range(), self.alloc(e), clock))
            },
            Some(ConcreteNode::TypeAppExpression) => {
                let e = self.parse_expr(node.child(0).unwrap())?;
                let ty = self.parse_type(node.child(3).unwrap())?;
                Ok(Expr::TypeApp(node.range(), self.alloc(e), ty))
            },
            Some(ConcreteNode::BinopExpression) => {
                let e1 = self.parse_expr(node.child(0).unwrap())?;
                let op = match self.node_text(node.child(1).unwrap()) {
                    "*" => Binop::FMul,
                    "/" => Binop::FDiv,
                    "+" => Binop::FAdd,
                    "-" => Binop::FSub,
                    ".<<." => Binop::Shl,
                    ".>>." => Binop::Shr,
                    ".&." => Binop::And,
                    ".^." => Binop::Xor,
                    ".|." => Binop::Or,
                    ".*." => Binop::IMul,
                    "./." => Binop::IDiv,
                    ".+." => Binop::IAdd,
                    ".-." => Binop::ISub,
                    ">" => Binop::FGt,
                    ">=" => Binop::FGe,
                    "<" => Binop::FLt,
                    "<=" => Binop::FLe,
                    "==" => Binop::FEq,
                    "!=" => Binop::FNe,
                    ".>." => Binop::IGt,
                    ".>=." => Binop::IGe,
                    ".<." => Binop::ILt,
                    ".<=." => Binop::ILe,
                    ".==." => Binop::IEq,
                    ".!=." => Binop::INe,
                    op => panic!("unknown binop \"{}\"", op)
                };
                let e2 = self.parse_expr(node.child(2).unwrap())?;
                Ok(Expr::Binop(node.range(), op, self.alloc(e1), self.alloc(e2)))
            },
            Some(_) =>
                Err(ParseError::ExpectedExpression(node.range())),
            None => 
                Err(ParseError::UnknownNodeType(node.range(), node.kind().into())),
        }
    }

    // TODO: add range information to Type?
    fn parse_type<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<Type, ParseError> {
        match self.parser.node_matcher.lookup(node.kind_id()) {
            Some(ConcreteNode::Type) =>
                self.parse_type(node.child(0).unwrap()),
            Some(ConcreteNode::WrapType) =>
                self.parse_type(node.child(1).unwrap()),
            Some(ConcreteNode::BaseType) =>
                Ok(match self.node_text(node) {
                    "sample" => Type::Sample,
                    "index" => Type::Index,
                    "unit" => Type::Unit,
                    base => panic!("unknown base type {base}"),
                }),
            Some(ConcreteNode::FunctionType) => {
                let ty1 = self.parse_type(node.child(0).unwrap())?;
                let ty2 = self.parse_type(node.child(2).unwrap())?;
                Ok(Type::Function(Box::new(ty1), Box::new(ty2)))
            },
            Some(ConcreteNode::StreamType) => {
                let clock = self.parse_clock(node.child(3).unwrap())?;
                let ty = self.parse_type(node.child(5).unwrap())?;
                Ok(Type::Stream(clock, Box::new(ty)))
            },
            Some(ConcreteNode::ProductType) => {
                let ty1 = self.parse_type(node.child(0).unwrap())?;
                let ty2 = self.parse_type(node.child(2).unwrap())?;
                Ok(Type::Product(Box::new(ty1), Box::new(ty2)))
            },
            Some(ConcreteNode::SumType) => {
                let ty1 = self.parse_type(node.child(0).unwrap())?;
                let ty2 = self.parse_type(node.child(2).unwrap())?;
                Ok(Type::Sum(Box::new(ty1), Box::new(ty2)))
            },
            Some(ConcreteNode::ArrayType) => {
                let ty = self.parse_type(node.child(1).unwrap())?;
                let size = self.parse_size(node.child(3).unwrap())?;
                Ok(Type::Array(Box::new(ty), size))
            },
            Some(ConcreteNode::LaterType) => {
                let clock = self.parse_clock(node.child(3).unwrap())?;
                let ty = self.parse_type(node.child(5).unwrap())?;
                Ok(Type::Later(clock, Box::new(ty)))
            },
            Some(ConcreteNode::BoxType) => {
                let ty = self.parse_type(node.child(1).unwrap())?;
                Ok(Type::Box(Box::new(ty)))
            },
            Some(ConcreteNode::ForallType) => {
                let x = self.identifier(node.child(1).unwrap());
                let k = self.parse_kind(node.child(3).unwrap())?;
                let ty = self.parse_type(node.child(5).unwrap())?;
                Ok(Type::Forall(x, k, Box::new(ty)))
            },
            Some(ConcreteNode::VarType) => {
                let x = self.identifier(node);
                Ok(Type::TypeVar(x))
            },
            Some(_) =>
                Err(ParseError::ExpectedType(node.range())),
            None =>
                Err(ParseError::UnknownNodeType(node.range(), node.kind().into())),
        }
    }

    fn parse_kind<'d>(&self, node: tree_sitter::Node<'d>) -> Result<Kind, ParseError> {
        match self.node_text(node) {
            "clock" => Ok(Kind::Clock),
            "type" => Ok(Kind::Type),
            kind => panic!("unknown kind {}", kind),
        }
    }

    fn parse_size<'d>(&self, node: tree_sitter::Node<'d>) -> Result<ArraySize, ParseError> {
        let text = self.node_text(node);
        match text.parse() {
            Ok(n) => Ok(ArraySize::from_const(n)),
            Err(_) => Err(ParseError::BadLiteral(node.range())),
        }
    }

    fn parse_clock<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<Clock, ParseError> {
        if node.child_count() == 1 {
            Ok(Clock { coeff: Ratio::from_integer(1), var: self.identifier(node.child(0).unwrap()) })
        } else {
            let coeff = self.parse_clock_coeff(node.child(0).unwrap())?;
            Ok(Clock { coeff, var: self.identifier(node.child(1).unwrap()) })
        }
    }

    fn parse_clock_coeff(&mut self, node: tree_sitter::Node<'_>) -> Result<Ratio<u32>, ParseError> {
        if node.child_count() == 0 {
            let n = self.node_text(node).parse().map_err(|_| ParseError::BadCoefficient(node.range()))?;
            Ok(Ratio::from_integer(n))
        } else {
            let n = self.node_text(node.child(0).unwrap()).parse().map_err(|_| ParseError::BadCoefficient(node.range()))?;
            let d = self.node_text(node.child(2).unwrap()).parse().map_err(|_| ParseError::BadCoefficient(node.range()))?;
            if d > 0 {
                Ok(Ratio::new(n, d))
            } else {
                Err(ParseError::BadCoefficient(node.range()))
            }
        }
    }
}
