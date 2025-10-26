from fuse import Program


def _sample_program() -> Program:
    return Program(
        """
X[i,k] = Input[i,k]
Y[i,j] = X[i,k] W[k,j]
export Y
""".strip()
    )


def test_program_explain_text_includes_index_table():
    prog = _sample_program()
    explanation = prog.explain()
    assert "[eq] Y" in explanation
    assert "LHS:" in explanation


def test_program_explain_json_structure():
    prog = _sample_program()
    payload = prog.explain(json=True)
    assert "equations" in payload
    equations = payload["equations"]
    assert equations, "expected at least one equation"
    target = next(eq for eq in equations if eq["name"] == "Y")
    assert target["index_summary"]["lhs"] == ["i", "j"]
    assert "rhs_only" in target["index_summary"]
