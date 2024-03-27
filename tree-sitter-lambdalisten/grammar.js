module.exports = grammar({
    name: 'lambdalisten',

    rules: {
        source_file: $ => $.expression,

        expression: $ => choice(
            $.wrap_expression,
            $.identifier,
            $.literal,
            $.application_expression,
            $.lambda_expression
        ),

        wrap_expression: $ => seq('(', $.expression, ')'),    

        identifier: $ => /[a-z]+/,

        literal: $ => /\d+/,

        application_expression: $ => prec.left(seq($.expression, $.expression)),

        lambda_expression: $ => prec.right(seq('\\', $.identifier, '.', $.expression))
    }
});
