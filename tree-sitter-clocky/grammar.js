module.exports = grammar({
    name: 'clocky',

    extras: $ => [
        /\s/, 
        $.comment
    ],

    rules: {
        source_file: $ => repeat1(choice($.top_level_def, $.top_level_let)),

        comment: $ => token(choice(
          seq('--', /(\\(.|\r?\n)|[^\\\n])*/),
          seq(
              '{-',
              /[^-]*-+([^\}-][^-]*-+)*/,
              '}'
          )
        )),

        top_level_def: $ => seq(
            'def',
            field('ident', $.identifier),
            ':',
            field('type', $.type),
            '=',
            field('body', $.expression),
            ';;'
        ),
        top_level_let: $ => seq(
            'let',
            field('ident', $.identifier),
            ':',
            field('type', $.type),
            '=',
            field('body', $.expression),
            ';;'
        ),

        expression: $ => choice(
            $.wrap_expression,
            $.identifier,
            $.literal,
            $.sample,
            $.application_expression,
            $.lambda_expression,
            $.lob_expression,
            $.force_expression,
            $.gen_expression,
            $.let_expression,
            $.annotate_expression,
            $.pair_expression,
            $.unpair_expression,
            $.inl_expression,
            $.inr_expression,
            $.case_expression,
            $.array_expression,
            $.ungen_expression,
            $.unit_expression,
            $.delay_expression,
            $.box_expression,
            $.unbox_expression,
            $.clockapp_expression,
            $.typeapp_expression,
            $.binop_expression
        ),

        wrap_expression: $ => seq('(', field('expr', $.expression), ')'),    

        identifier: $ => /[a-z][a-z0-9_]*/,

        literal: $ => choice(/\d+/, seq('0x', /[\da-fA-F]+/)),

        sample: $ => /[-+]?\d+\.\d*/,

        application_expression: $ => prec.left(10, seq(
            field('func', $.expression),
            field('arg', $.expression)
        )),

        lambda_expression: $ => prec.right(seq(
            '\\',
            field('binder', $.identifier),
            '.',
            field('body', $.expression)
        )),

        lob_expression: $ => prec.right(seq(
            '&',
            '^',
            '(',
            field('clock', $.clock),
            ')',
            field('binder', $.identifier),
            '.',
            field('body', $.expression)
        )),

        force_expression: $ => prec(11, seq('!', field('expr', $.expression))),

        gen_expression: $ => prec.right(seq(
            field('head', $.expression),
            '::',
            field('tail', $.expression)
        )),

        let_expression: $ => prec.left(-1, seq(
            'let',
            field('binder', $.identifier),
            optional(seq(':', field('type', $.type))),
            '=',
            field('bound', $.expression),
            'in',
            field('body', $.expression)
        )),

        annotate_expression: $ => seq(field('expr', $.expression), ':', field('type', $.type)),

        pair_expression: $ => seq(
            '(',
            field('left', $.expression),
            ',',
            field('right', $.expression),
            ')'
        ),

        unpair_expression: $ => prec.left(-1, seq(
            'let',
            '(',
            field('binderleft', $.identifier),
            ',',
            field('binderright', $.identifier),
            ')',
            '=',
            field('bound', $.expression),
            'in',
            field('body', $.expression)
        )),

        inl_expression: $ => prec(11, seq('inl', field('expr', $.expression))),

        inr_expression: $ => prec(11, seq('inr', field('expr', $.expression))),

        case_expression: $ => seq(
            'case',
            field('scrutinee', $.expression),
            '{',
            'inl',
            field('binderleft', $.identifier),
            '=>',
            field('bodyleft', $.expression),
            '|',
            'inr',
            field('binderright', $.identifier),
            '=>',
            field('bodyright', $.expression),
            '}'
        ),

        array_expression: $ => choice(seq('[', ']'), seq('[', field('inner', $.array_inner), ']')),

        array_inner: $ => seq(repeat(seq(field('expr', $.expression), ',')), field('expr', $.expression)),

        ungen_expression: $ => prec(11, seq('%', field('expr', $.expression))),

        unit_expression: $ => '()',

        delay_expression: $ => prec(11, seq('`', field('expr', $.expression))),

        box_expression: $ => prec(11, seq('box', field('expr', $.expression))),

        unbox_expression: $ => prec(11, seq('unbox', field('expr', $.expression))),

        clockapp_expression: $ => prec.left(seq(
            field('expr', $.expression),
            '@',
            '(',
            field('clock', $.clock),
            ')'
        )),

        typeapp_expression: $ => prec.left(seq(
            field('expr', $.expression),
            '$',
            '(',
            field('type', $.type),
            ')'
        )),

        binop_expression: $ => choice(
            prec.left(2, seq(field('left', $.expression), field('op', '*'), field('right', $.expression))),
            prec.left(2, seq(field('left', $.expression), field('op', '.*.'), field('right', $.expression))),
            prec.left(2, seq(field('left', $.expression), field('op', '/'), field('right', $.expression))),
            prec.left(2, seq(field('left', $.expression), field('op', './.'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '+'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '.+.'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '-'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '.-.'), field('right', $.expression))),
            // TODO: fix these precedences
            prec.left(1, seq(field('left', $.expression), field('op', '.<<.'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '.>>.'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '.&.'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '.^.'), field('right', $.expression))),
            prec.left(1, seq(field('left', $.expression), field('op', '.|.'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '>'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '>='), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '<'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '<='), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '=='), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '!='), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '.>.'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '.>=.'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '.<.'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '.<=.'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '.==.'), field('right', $.expression))),
            prec.left(0, seq(field('left', $.expression), field('op', '.!=.'), field('right', $.expression))),
        ),

        type: $ => choice(
            $.wrap_type,
            $.base_type,
            $.function_type,
            $.stream_type,
            $.product_type,
            $.sum_type,
            $.array_type,
            $.later_type,
            $.box_type,
            $.forall_type,
            $.var_type
        ),

        wrap_type: $ => seq('(', field('type', $.type), ')'),

        base_type: $ => choice(
            'sample',
            'index',
            'unit'
        ),

        function_type: $ => prec.right(seq(field('arg', $.type), '->', field('ret', $.type))),

        stream_type: $ => prec(3, seq('~', '^', '(', field('clock', $.clock), ')', field('type', $.type))),

        product_type: $ => prec.right(2, seq(field('left', $.type), '*', field('right', $.type))),

        sum_type: $ => prec.right(1, seq(field('left', $.type), '+', field('right', $.type))),

        array_type: $ => seq('[', field('type', $.type), ';', field('size', $.size), ']'),

        later_type: $ => prec(3, seq('|>', '^', '(', field('clock', $.clock), ')', field('type', $.type))),

        box_type: $ => prec(3, seq('[]', field('type', $.type))),

        size: $ => /[\d]+/,

        // TODO: put the parentheses in here, and make it so that bare identifiers don't need them
        clock: $ => choice(
            field('ident', $.identifier),
            seq(field('coeff', $.clock_coeff), field('ident', $.identifier))
        ),

        clock_coeff: $ => choice(/[\d]+/, seq(/[\d]+/, "/", /[\d]+/)),

        forall_type: $ => prec.right(seq(
            'for',
            field('binder', $.identifier),
            ':',
            field('kind', $.kind),
            '.',
            field('type', $.type)
        )),

        var_type: $ => $.identifier,

        kind: $ => choice('clock', 'type')
    }
});
