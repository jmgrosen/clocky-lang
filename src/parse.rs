use string_interner::DefaultStringInterner;
use typed_arena::Arena;
use num::rational::Ratio;

use crate::{expr::{Binop, Expr, SourceFile, Symbol, TopLevelDef, TopLevelDefBody, TopLevelDefKind, Value}, typing::{Type, ArraySize, Clock, Kind}};

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

macro_rules! count {
    ( ) => { 0 };
    ( $x:ident ) => { 1 };
    ( $x:ident, $($y:ident),* ) => {1 + count!($($y),*) }
}

macro_rules! make_field_enum {
    ($enum_name:ident { $($rust_name:ident : $ts_name:ident),* } with matcher $matcher_name:ident) => {
        #[derive(PartialEq, Eq, Debug, Copy, Clone)]
        #[repr(u16)]
        enum $enum_name {
            $( $rust_name ),*
        }

        pub struct $matcher_name {
            field_id_table: [u16; count!($($rust_name),*)],
        }

        impl $matcher_name {
            fn new(lang: &tree_sitter::Language) -> $matcher_name {
                $matcher_name {
                    field_id_table: [
                        $( lang.field_id_for_name(stringify!($ts_name)).expect(concat!("could not find field name ", stringify!($ts_name))) ),*
                    ]
                }
            }

            fn lookup(&self, field: $enum_name) -> u16 {
                self.field_id_table[field as usize]
            }
        }
    };
}

// why isn't this information in the generated bindings...?
make_node_enum!(ConcreteNode {
    SourceFile: source_file,
    TopLevelDef: top_level_def,
    TopLevelLet: top_level_let,
    TopLevelClock: top_level_clock,
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
    ExIntro: ex_intro,
    ExElim: ex_elim,
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
    ExType: ex_type,
    Kind: kind
} with matcher ConcreteNodeMatcher);

make_field_enum!(Field {
    Ident: ident,
    Type: type,
    Expr: expr,
    Func: func,
    Arg: arg,
    Frequency: frequency,
    Binder: binder,
    Body: body,
    Clock: clock,
    Head: head,
    Tail: tail,
    Bound: bound,
    Scrutinee: scrutinee,
    BinderLeft: binderleft,
    BodyLeft: bodyleft,
    BinderRight: binderright,
    BodyRight: bodyright,
    Inner: inner,
    Left: left,
    Right: right,
    Op: op,
    Ret: ret,
    Size: size,
    Coeff: coeff,
    Kind: kind,
    BinderClock: binderclock,
    BinderExpr: binderexpr
} with matcher ConcreteFieldMatcher);

pub struct Parser<'a, 'b> {
    parser: tree_sitter::Parser,
    node_matcher: ConcreteNodeMatcher,
    field_matcher: ConcreteFieldMatcher,
    pub interner: &'a mut DefaultStringInterner,
    pub arena: &'b Arena<Expr<'b, tree_sitter::Range>>,
}

impl<'a, 'b> Parser<'a, 'b> {
    pub fn new(interner: &'a mut DefaultStringInterner, arena: &'b Arena<Expr<'b, tree_sitter::Range>>) -> Parser<'a, 'b> {
        let mut parser = tree_sitter::Parser::new();
        let lang = tree_sitter_clocky::language();
        let node_matcher = ConcreteNodeMatcher::new(&lang);
        let field_matcher = ConcreteFieldMatcher::new(&lang);
        parser.set_language(lang).expect("Error loading clocky grammar");

        Parser {
            parser,
            node_matcher,
            field_matcher,
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

    fn field_opt<'d>(&self, node: tree_sitter::Node<'d>, field: Field) -> Option<tree_sitter::Node<'d>> {
        node.child_by_field_id(self.parser.field_matcher.lookup(field))
    }

    fn field<'d>(&self, node: tree_sitter::Node<'d>, field: Field) -> tree_sitter::Node<'d> {
        self.field_opt(node, field)
            .unwrap_or_else(|| panic!("node {:?} did not have field {:?}", node, field))
    }

    fn parse_file<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<SourceFile<'b, tree_sitter::Range>, ParseError> {
        let Some(ConcreteNode::SourceFile) = self.parser.node_matcher.lookup(node.kind_id()) else {
            return Err(ParseError::UhhhhhhWhat(node.range(), "you didn't pass me a file".to_string()));
        };

        let mut defs = Vec::new();
        let mut cur = node.walk();
        cur.goto_first_child();
        loop {
            let node = cur.node();
            if !node.is_extra() {
                defs.push(self.parse_top_level_decl(node)?);
            }
            if !cur.goto_next_sibling() {
                break;
            }
        }

        Ok(SourceFile { defs })
    }

    fn parse_top_level_decl<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<TopLevelDef<'b, tree_sitter::Range>, ParseError> {
        let body = match self.parser.node_matcher.lookup(node.kind_id()) {
            Some(ConcreteNode::TopLevelDef) => {
                let type_ = self.parse_type(self.field(node, Field::Type))?;
                let expr = self.parse_expr(self.field(node, Field::Body))?;
                TopLevelDefBody::Def {
                    kind: TopLevelDefKind::Def,
                    type_,
                    expr: self.alloc(expr),
                }
            },
            Some(ConcreteNode::TopLevelLet) => {
                let type_ = self.parse_type(self.field(node, Field::Type))?;
                let expr = self.parse_expr(self.field(node, Field::Body))?;
                TopLevelDefBody::Def {
                    kind: TopLevelDefKind::Let,
                    type_,
                    expr: self.alloc(expr),
                }
            },
            Some(ConcreteNode::TopLevelClock) => TopLevelDefBody::Clock {
                freq: self.parse_freq(self.field(node, Field::Frequency))?,
            },
            _ => return Err(ParseError::UhhhhhhWhat(node.range(), "expected a top-level let here".to_string()))
        };

        Ok(TopLevelDef {
            name: self.identifier(self.field(node, Field::Ident)),
            range: node.range(),
            body,
        })
    }

    fn parse_freq<'d>(&self, node: tree_sitter::Node<'d>) -> Result<f32, ParseError> {
        let freq_text = self.node_text(node);
        freq_text.parse().map_err(|_| ParseError::BadLiteral(node.range()))
    }

    fn parse_expr<'d>(&mut self, node: tree_sitter::Node<'d>) -> Result<Expr<'b, tree_sitter::Range>, ParseError> {
        // TODO: use a TreeCursor instead
        match self.parser.node_matcher.lookup(node.kind_id()) {
            Some(ConcreteNode::Expression) =>
                self.parse_expr(node.child(0).unwrap()),
            Some(ConcreteNode::WrapExpression) =>
                // the literals are included in the children indices
                self.parse_expr(self.field(node, Field::Expr)),
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
                let e1 = self.parse_expr(self.field(node, Field::Func))?;
                let e2 = self.parse_expr(self.field(node, Field::Arg))?;
                Ok(Expr::App(node.range(), self.parser.arena.alloc(e1), self.parser.arena.alloc(e2)))
            },
            Some(ConcreteNode::LambdaExpression) => {
                let x = self.parser.interner.get_or_intern(self.node_text(self.field(node, Field::Binder)));
                let e = self.parse_expr(self.field(node, Field::Body))?;
                Ok(Expr::Lam(node.range(), x, self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::LobExpression) => {
                let clock = self.parse_clock(self.field(node, Field::Clock))?;
                let x = self.parser.interner.get_or_intern(self.node_text(self.field(node, Field::Binder)));
                let e = self.parse_expr(self.field(node, Field::Body))?;
                Ok(Expr::Lob(node.range(), clock, x, self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::ForceExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::Adv(node.range(), self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::GenExpression) => {
                let e1 = self.parse_expr(self.field(node, Field::Head))?;
                let e2 = self.parse_expr(self.field(node, Field::Tail))?;
                Ok(Expr::Gen(node.range(), self.parser.arena.alloc(e1), self.parser.arena.alloc(e2)))
            },
            Some(ConcreteNode::LetExpression) => {
                let x = self.parser.interner.get_or_intern(self.node_text(self.field(node, Field::Binder)));
                let ty = self.field_opt(node, Field::Type).map(|t| self.parse_type(t)).transpose()?;
                let e1 = self.parse_expr(self.field(node, Field::Bound))?;
                let e2 = self.parse_expr(self.field(node, Field::Body))?;
                Ok(Expr::LetIn(node.range(), x, ty, self.parser.arena.alloc(e1), self.parser.arena.alloc(e2)))
            },
            Some(ConcreteNode::AnnotateExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                let ty = self.parse_type(self.field(node, Field::Type))?;
                Ok(Expr::Annotate(node.range(), self.parser.arena.alloc(e), ty))
            },
            Some(ConcreteNode::PairExpression) => {
                let e1 = self.parse_expr(self.field(node, Field::Left))?;
                let e2 = self.parse_expr(self.field(node, Field::Right))?;
                Ok(Expr::Pair(node.range(), self.alloc(e1), self.alloc(e2)))
            },
            Some(ConcreteNode::UnPairExpression) => {
                let x1 = self.identifier(self.field(node, Field::BinderLeft));
                let x2 = self.identifier(self.field(node, Field::BinderRight));
                let e0 = self.parse_expr(self.field(node, Field::Bound))?;
                let e = self.parse_expr(self.field(node, Field::Body))?;
                Ok(Expr::UnPair(node.range(), x1, x2, self.alloc(e0), self.alloc(e)))
            },
            Some(ConcreteNode::InLExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::InL(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::InRExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::InR(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::CaseExpression) => {
                let e0 = self.parse_expr(self.field(node, Field::Scrutinee))?;
                let x1 = self.identifier(self.field(node, Field::BinderLeft));
                let e1 = self.parse_expr(self.field(node, Field::BodyLeft))?;
                let x2 = self.identifier(self.field(node, Field::BinderRight));
                let e2 = self.parse_expr(self.field(node, Field::BodyRight))?;
                Ok(Expr::Case(node.range(), self.alloc(e0), x1, self.alloc(e1), x2, self.alloc(e2)))
            },
            Some(ConcreteNode::ArrayExpression) => {
                Ok(Expr::Array(node.range(), match self.field_opt(node, Field::Inner) {
                    Some(array_inner) => {
                        let mut es = Vec::with_capacity((array_inner.child_count() + 1) / 2);
                        let mut cur = array_inner.walk();
                        let expr_field = self.parser.field_matcher.lookup(Field::Expr);
                        for elem_node in array_inner.children_by_field_id(expr_field, &mut cur) {
                            let elem = self.parse_expr(elem_node)?;
                            es.push(self.alloc(elem));
                        }
                        es.into()
                    },
                    None =>
                        [].into(),
                }))
            },
            Some(ConcreteNode::UnGenExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::UnGen(node.range(), self.parser.arena.alloc(e)))
            },
            Some(ConcreteNode::UnitExpression) =>
                Ok(Expr::Val(node.range(), Value::Unit)),
            Some(ConcreteNode::DelayExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::Delay(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::BoxExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::Box(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::UnboxExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::Unbox(node.range(), self.alloc(e)))
            },
            Some(ConcreteNode::ClockAppExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                let clock = self.parse_clock(self.field(node, Field::Clock))?;
                Ok(Expr::ClockApp(node.range(), self.alloc(e), clock))
            },
            Some(ConcreteNode::TypeAppExpression) => {
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                let ty = self.parse_type(self.field(node, Field::Type))?;
                Ok(Expr::TypeApp(node.range(), self.alloc(e), ty))
            },
            Some(ConcreteNode::BinopExpression) => {
                let e1 = self.parse_expr(self.field(node, Field::Left))?;
                let op = match self.node_text(self.field(node, Field::Op)) {
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
                let e2 = self.parse_expr(self.field(node, Field::Right))?;
                Ok(Expr::Binop(node.range(), op, self.alloc(e1), self.alloc(e2)))
            },
            Some(ConcreteNode::ExIntro) => {
                let c = self.parse_clock(self.field(node, Field::Clock))?;
                let e = self.parse_expr(self.field(node, Field::Expr))?;
                Ok(Expr::ExIntro(node.range(), c, self.alloc(e)))
            },
            Some(ConcreteNode::ExElim) => {
                let c = self.identifier(self.field(node, Field::BinderClock));
                let x = self.identifier(self.field(node, Field::BinderExpr));
                let e1 = self.parse_expr(self.field(node, Field::Bound))?;
                let e2 = self.parse_expr(self.field(node, Field::Body))?;
                Ok(Expr::ExElim(node.range(), c, x, self.alloc(e1), self.alloc(e2)))
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
                self.parse_type(self.field(node, Field::Type)),
            Some(ConcreteNode::BaseType) =>
                Ok(match self.node_text(node) {
                    "sample" => Type::Sample,
                    "index" => Type::Index,
                    "unit" => Type::Unit,
                    base => panic!("unknown base type {base}"),
                }),
            Some(ConcreteNode::FunctionType) => {
                let ty1 = self.parse_type(self.field(node, Field::Arg))?;
                let ty2 = self.parse_type(self.field(node, Field::Ret))?;
                Ok(Type::Function(Box::new(ty1), Box::new(ty2)))
            },
            Some(ConcreteNode::StreamType) => {
                let clock = self.parse_clock(self.field(node, Field::Clock))?;
                let ty = self.parse_type(self.field(node, Field::Type))?;
                Ok(Type::Stream(clock, Box::new(ty)))
            },
            Some(ConcreteNode::ProductType) => {
                let ty1 = self.parse_type(self.field(node, Field::Left))?;
                let ty2 = self.parse_type(self.field(node, Field::Right))?;
                Ok(Type::Product(Box::new(ty1), Box::new(ty2)))
            },
            Some(ConcreteNode::SumType) => {
                let ty1 = self.parse_type(self.field(node, Field::Left))?;
                let ty2 = self.parse_type(self.field(node, Field::Right))?;
                Ok(Type::Sum(Box::new(ty1), Box::new(ty2)))
            },
            Some(ConcreteNode::ArrayType) => {
                let ty = self.parse_type(self.field(node, Field::Type))?;
                let size = self.parse_size(self.field(node, Field::Size))?;
                Ok(Type::Array(Box::new(ty), size))
            },
            Some(ConcreteNode::LaterType) => {
                let clock = self.parse_clock(self.field(node, Field::Clock))?;
                let ty = self.parse_type(self.field(node, Field::Type))?;
                Ok(Type::Later(clock, Box::new(ty)))
            },
            Some(ConcreteNode::BoxType) => {
                let ty = self.parse_type(self.field(node, Field::Type))?;
                Ok(Type::Box(Box::new(ty)))
            },
            Some(ConcreteNode::ForallType) => {
                let x = self.identifier(self.field(node, Field::Binder));
                let k = self.parse_kind(self.field(node, Field::Kind))?;
                let ty = self.parse_type(self.field(node, Field::Type))?;
                Ok(Type::Forall(x, k, Box::new(ty)))
            },
            Some(ConcreteNode::VarType) => {
                let x = self.identifier(node);
                Ok(Type::TypeVar(x))
            },
            Some(ConcreteNode::ExType) => {
                let c = self.identifier(self.field(node, Field::Binder));
                let ty = self.parse_type(self.field(node, Field::Type))?;
                Ok(Type::Exists(c, Box::new(ty)))
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
        let coeff = if let Some(coeff_node) = self.field_opt(node, Field::Coeff) {
            self.parse_clock_coeff(coeff_node)?
        } else {
            Ratio::from_integer(1)
        };
        let var = self.identifier(self.field(node, Field::Ident));
        Ok(Clock { coeff, var })
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
