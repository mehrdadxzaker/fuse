from fuse.core.ast import Block, Equation, Let, pretty_print_old_style
from fuse.core.parser_expr import parse_program


def test_parse_and_pretty_print_smoke():
    src = (
        'import "data.bin" as data;\n'
        'param N: int = 128;\n'
        'axis i;\n'
        'const K = 3;\n'
        'fn foo(x) { let y = x * 2; A[i] = y + 1; }\n'
        'let t = 1 + 2 * 3;\n'
        'A[i] = t ? foo(t) : 0;\n'
    )

    prog = parse_program(src)

    # Basic structural assertions
    assert any(isinstance(st, Let) for st in prog.statements)
    assert any(isinstance(st, Equation) for st in prog.statements)
    assert prog.fns and prog.fns[0].name == "foo"
    assert isinstance(prog.fns[0].body, Block)

    rendered = pretty_print_old_style(prog)
    # Pretty-print keeps legacy style lines
    assert "param N: int = 128;" in rendered
    assert "const K = 3;" in rendered
    assert "let t = 1 + 2 * 3;" in rendered
    assert "A[i] = t ? foo(t) : 0;" in rendered
