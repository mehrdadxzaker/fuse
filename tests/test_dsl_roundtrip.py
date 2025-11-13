import pytest

from fuse.core.ast import pretty_print_old_style
from fuse.core.parser_expr import parse_program


@pytest.mark.parametrize(
    "src",
    [
        "param D:int = 8; axis i; axis j; const Z = 2; let x[i] = Z; y[i,j] = x[i] * x[i] + 1; export y;",
        "fn dot(a[x], b[x]) -> s { s = a[x] * b[x]; } sim[i,j] = dot(E[i,d], E[j,d]); export sim;",
        "score[i] = reduce(sum, d) x[i,d] * x[i,d] ** 0.5; export score;",
        "z[i] = select(m[i], a[i], b[i]); export z;",
        "z[i] when m[i] = a[i]; z[i] when !m[i] = b[i]; export z;",
        "score[i] = case { m[i] -> a[i]; default -> b[i]; }; export score;",
        "total[i] = reduce(sum, j)  A[i,j] * B[j]; export total;",
    ],
)
def test_roundtrip_pretty_parse(src):
    prog1 = parse_program(src)
    text1 = pretty_print_old_style(prog1)
    prog2 = parse_program(text1)
    text2 = pretty_print_old_style(prog2)
    assert text1 == text2
