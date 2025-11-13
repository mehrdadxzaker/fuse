from fuse.core.ast_lowering import lower_to_ir
from fuse.core.ir import FuncCall, TensorRef, Term
from fuse.core.parser_expr import parse_program


def _find_eq(ir, name):
    return [eq for eq in ir.equations if eq.lhs.name == name]


def test_let_and_reduce_lowering():
    src = "let sim[u,v] = Emb[u,d] * Emb[v,d];\nscore[u] = reduce(sum, v) sim[u,v] * w[v];\n"
    ast_prog = parse_program(src)
    ir = lower_to_ir(ast_prog)

    sim_eqs = _find_eq(ir, "sim")
    assert len(sim_eqs) == 1
    sim_eq = sim_eqs[0]
    assert sim_eq.lhs.indices == ["u", "v"]
    assert isinstance(sim_eq.rhs, Term)
    names = [f.name for f in sim_eq.rhs.factors if isinstance(f, TensorRef)]
    assert names.count("Emb") == 2

    # Should create a reduction temp feeding into score
    score_eqs = _find_eq(ir, "score")
    assert score_eqs, "expected score equation"
    # There should be a __red temporary defined
    red_eqs = [eq for eq in ir.equations if eq.lhs.name.startswith("__red")]
    assert red_eqs, "expected a reduction temporary"
    # Reduction temp should project with sum
    assert all(eq.projection == "sum" for eq in red_eqs)


def test_when_guard_lowering():
    src = "score[u] when risky[u] = score_hi[u];\nscore[u] when !risky[u] = score_lo[u];\n"
    ast_prog = parse_program(src)
    ir = lower_to_ir(ast_prog)
    score_eqs = _find_eq(ir, "score")
    assert len(score_eqs) == 2
    # First guarded eq should multiply by risky
    lhs_u = score_eqs[0].lhs.indices
    assert lhs_u == ["u"]
    rhs0 = score_eqs[0].rhs
    assert isinstance(rhs0, Term)
    factors0 = rhs0.factors
    assert any(isinstance(f, TensorRef) and f.name == "risky" for f in factors0)
    # Second guarded eq should include a case() for !risky
    rhs1 = score_eqs[1].rhs
    assert isinstance(rhs1, Term)
    assert any(isinstance(f, FuncCall) and f.name == "case" for f in rhs1.factors)


def test_power_and_div_lowering():
    src = "let z[i,d] = x[i,d] / (y[d] ** 0.5);\nnorm[i] = reduce(sum, d) (z[i,d] * z[i,d]);\n"
    ast_prog = parse_program(src)
    ir = lower_to_ir(ast_prog)
    # First equation should produce a pow() in the term
    z_eq = _find_eq(ir, "z")[0]
    assert isinstance(z_eq.rhs, Term)
    assert any(isinstance(f, FuncCall) and f.name == "pow" for f in z_eq.rhs.factors)
    # Second equation should create a reduction temp
    red_eqs = [eq for eq in ir.equations if eq.lhs.name.startswith("__red")]
    assert red_eqs
