module.exports = grammar({
    name: 'lambdalisten',

    rules: {
        source_file: $ => $.expression,

        expression: $ => choice(
            $.wrap_expression,
            $.identifier,
            $.literal,
            $.sample,
            $.application_expression,
            $.lambda_expression,
            $.lob_expression,
            $.force_expression,
            $.gen_expression
        ),

        wrap_expression: $ => seq('(', $.expression, ')'),    

        identifier: $ => /[a-z]+/,

        literal: $ => /\d+/,

        sample: $ => /-?\d+.\d*/,

        application_expression: $ => prec.left(seq($.expression, $.expression)),

        lambda_expression: $ => prec.right(seq('\\', $.identifier, '.', $.expression)),

        lob_expression: $ => prec.right(seq('&', $.identifier, '.', $.expression)),

        force_expression: $ => prec(2, seq('!', $.expression)),

        gen_expression: $ => prec.right(seq($.expression, '::', $.expression))
    }
});
