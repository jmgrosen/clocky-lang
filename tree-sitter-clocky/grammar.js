module.exports = grammar({
    name: 'clocky',

    rules: {
        source_file: $ => repeat1($.top_level_def),

        top_level_def: $ => seq('def', $.identifier, ':', $.type, '=', $.expression),

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

        wrap_expression: $ => seq('(', $.expression, ')'),    

        identifier: $ => /[a-z][a-z0-9_]*/,

        literal: $ => choice(/\d+/, seq('0x', /[\da-fA-F]+/)),

        sample: $ => /[-+]?\d+\.\d*/,

        application_expression: $ => prec.left(10, seq($.expression, $.expression)),

        lambda_expression: $ => prec.right(seq('\\', $.identifier, '.', $.expression)),

        lob_expression: $ => prec.right(seq('&', '^', '(', $.clock, ')', $.identifier, '.', $.expression)),

        force_expression: $ => prec(11, seq('!', $.expression)),

        gen_expression: $ => prec.right(seq($.expression, '::', $.expression)),

        let_expression: $ => prec.left(-1, seq('let', $.identifier, optional(seq(':', $.type)), '=', field('bound', $.expression), 'in', field('body', $.expression))),

        annotate_expression: $ => seq($.expression, ':', $.type),

        pair_expression: $ => seq('(', $.expression, ',', $.expression, ')'),

        unpair_expression: $ => prec.left(-1, seq('let', '(', $.identifier, ',', $.identifier, ')', '=', field('bound', $.expression), 'in', field('body', $.expression))),

        inl_expression: $ => prec(11, seq('inl', $.expression)),

        inr_expression: $ => prec(11, seq('inr', $.expression)),

        case_expression: $ => seq('case', $.expression, '{', 'inl', $.identifier, '=>', $.expression, '|', 'inr', $.identifier, '=>', $.expression, '}'),

        array_expression: $ => choice(seq('[', ']'), seq('[', $.array_inner, ']')),

        array_inner: $ => seq(repeat(seq($.expression, ',')), $.expression),

        ungen_expression: $ => prec(11, seq('%', $.expression)),

        unit_expression: $ => '()',

        delay_expression: $ => prec(11, seq('`', $.expression)),

        box_expression: $ => prec(11, seq('box', $.expression)),

        unbox_expression: $ => prec(11, seq('unbox', $.expression)),

        clockapp_expression: $ => prec.left(seq($.expression, '@', '(', $.clock, ')')),

        typeapp_expression: $ => prec.left(seq($.expression, '$', '(', $.type, ')')),

        binop_expression: $ => choice(
            prec.left(2, seq($.expression, '*', $.expression)),
            prec.left(2, seq($.expression, '.*.', $.expression)),
            prec.left(2, seq($.expression, '/', $.expression)),
            prec.left(2, seq($.expression, './.', $.expression)),
            prec.left(1, seq($.expression, '+', $.expression)),
            prec.left(1, seq($.expression, '.+.', $.expression)),
            prec.left(1, seq($.expression, '-', $.expression)),
            prec.left(1, seq($.expression, '.-.', $.expression)),
            // TODO: fix these precedences
            prec.left(1, seq($.expression, '.<<.', $.expression)),
            prec.left(1, seq($.expression, '.>>.', $.expression)),
            prec.left(1, seq($.expression, '.&.', $.expression)),
            prec.left(1, seq($.expression, '.^.', $.expression)),
            prec.left(1, seq($.expression, '.|.', $.expression)),
            prec.left(0, seq($.expression, '>', $.expression)),
            prec.left(0, seq($.expression, '>=', $.expression)),
            prec.left(0, seq($.expression, '<', $.expression)),
            prec.left(0, seq($.expression, '<=', $.expression)),
            prec.left(0, seq($.expression, '==', $.expression)),
            prec.left(0, seq($.expression, '!=', $.expression)),
            prec.left(0, seq($.expression, '.>.', $.expression)),
            prec.left(0, seq($.expression, '.>=.', $.expression)),
            prec.left(0, seq($.expression, '.<.', $.expression)),
            prec.left(0, seq($.expression, '.<=.', $.expression)),
            prec.left(0, seq($.expression, '.==.', $.expression)),
            prec.left(0, seq($.expression, '.!=.', $.expression)),
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

        wrap_type: $ => seq('(', $.type, ')'),

        base_type: $ => choice(
            'sample',
            'index',
            'unit'
        ),

        function_type: $ => prec.right(seq($.type, '->', $.type)),

        stream_type: $ => prec(3, seq('~', '^', '(', $.clock, ')', $.type)),

        product_type: $ => prec.right(2, seq($.type, '*', $.type)),

        sum_type: $ => prec.right(1, seq($.type, '+', $.type)),

        array_type: $ => seq('[', $.type, ';', $.size, ']'),

        later_type: $ => prec(3, seq('|>', '^', '(', $.clock, ')', $.type)),

        box_type: $ => prec(3, seq('[]', $.type)),

        size: $ => /[\d]+/,

        // TODO: put the parentheses in here, and make it so that bare identifiers don't need them
        clock: $ => choice($.identifier, seq($.clock_coeff, $.identifier)),

        clock_coeff: $ => choice(/[\d]+/, seq(/[\d]+/, "/", /[\d]+/)),

        forall_type: $ => prec.right(seq('for', $.identifier, ':', $.kind, '.', $.type)),

        var_type: $ => $.identifier,

        kind: $ => choice('clock', 'type')
    }
});
