from fuse.core.ast_lowering import lower_to_ir
from fuse.core.ir import FuncCall, TensorRef, Term
from fuse.core.parser_expr import parse_program


def _eqs_for(ir, name):
    return [eq for eq in ir.equations if eq.lhs.name == name]


def test_fn_inline_dot_and_param_const_fold():
    src = (
        "param D: int = 128;\n"
        "fn dot(a[x], b[x]) -> s[] { s[] = a[x] * b[x]; }\n"
        "sim[i,j] = dot(Emb[i,d], Emb[j,d]);\n"
        "top[i] = k_select(sim[i,j], k=D/8);\n"
    )
    ast_prog = parse_program(src)
    ir = lower_to_ir(ast_prog)

    # Expect a function expansion temp for the return value with dims [i, j]
    fn_temps = [eq for eq in ir.equations if eq.lhs.name.startswith("__fn")]
    assert fn_temps, "expected inlined function equations"
    ret_eq = fn_temps[-1]
    assert ret_eq.lhs.indices == ["i", "j"]
    assert isinstance(ret_eq.rhs, Term)
    assert sum(1 for f in ret_eq.rhs.factors if isinstance(f, TensorRef) and f.name == "Emb") == 2

    # The k argument should be constant-folded to 16
    k_calls = [
        eq for eq in ir.equations if isinstance(eq.rhs, FuncCall) and eq.rhs.name == "k_select"
    ]
    assert k_calls, "expected k_select call in IR"
    k_kwargs = k_calls[0].rhs.kwargs  # type: ignore[attr-defined]
    assert isinstance(k_kwargs.get("k"), (int, float)) and int(k_kwargs.get("k")) == 16
