from fuse.core.ast import Call, Equation, Let
from fuse.core.macro_rewrite import expand_macros
from fuse.core.parser_expr import parse_program


def test_softmax_macro_expands_to_call():
    src = 'let y[i,j] = @softmax(x[i,j], axis=j);'
    ast_prog = parse_program(src)
    ast_prog2 = expand_macros(ast_prog)
    stmt = ast_prog2.statements[0]
    assert isinstance(stmt, Let)
    call = stmt.value
    assert isinstance(call, Call)
    assert call.name == 'softmax'
    assert call.kwargs.get('axis') == 'j'


def test_masked_softmax_macro():
    src = 'z[i,j] = @softmax(x[i,j], axis=j, mask=m[i,j]);'
    ast_prog = parse_program(src)
    ast_prog2 = expand_macros(ast_prog)
    stmt = ast_prog2.statements[0]
    assert isinstance(stmt, Equation)
    call = stmt.rhs
    assert isinstance(call, Call)
    assert call.name == 'masked_softmax'
    assert 'mask' in call.kwargs


def test_layer_norm_macro():
    src = 'y[i,d] = @layer_norm(x[i,d], axis=d, eps=1e-5);'
    ast_prog = parse_program(src)
    ast_prog2 = expand_macros(ast_prog)
    call = ast_prog2.statements[0].rhs
    assert isinstance(call, Call)
    assert call.name in ('layernorm',)

