import textwrap

import pytest

from fuse import Program

SHAPE_ERROR_CASES = [
    (
        "missing_axis_in_simple_assignment",
        """
        Out[i] = In[j]
        """,
        ["Out: RHS joins on {j}, projects {j}, but LHS expects {i}", "missing indices i"],
    ),
    (
        "scalar_missing_projection",
        """
        Out[] = In[i]
        """,
        [
            "Out: RHS joins on {i}, projects {i}, but LHS expects {∅}",
            "unmatched RHS indices i from In",
        ],
    ),
    (
        "missing_second_lhs_axis",
        """
        Out[i,j] = In[i]
        """,
        ["Out: RHS joins on {i}, projects {∅}, but LHS expects {i, j}", "missing indices j"],
    ),
    (
        "stray_axis_single_factor",
        """
        Out[i] = In[i] Extra[k]
        """,
        [
            "Out: RHS joins on {i, k}, projects {k}, but LHS expects {i}",
            "unmatched RHS indices k from Extra",
        ],
    ),
    (
        "two_stray_axes",
        """
        Out[i] = In[i] Extra[k,l]
        """,
        ["Out: RHS joins on {i, k, l}, projects {k, l}, but LHS expects {i}", "Extra"],
    ),
    (
        "lhs_axis_missing_completely",
        """
        Out[i] = In[k]
        """,
        ["Out: RHS joins on {k}, projects {k}, but LHS expects {i}", "missing indices i"],
    ),
    (
        "stray_axis_from_mixed_factor",
        """
        Out[i] = In[i] Mixed[i,k]
        """,
        [
            "Out: RHS joins on {i, k}, projects {k}, but LHS expects {i}",
            "unmatched RHS indices k from Mixed",
        ],
    ),
    (
        "single_stray_in_multi_factor_product",
        """
        Out[i] = In[i] First[k] Second[j]
        """,
        ["Out: RHS joins on {i, j, k}, projects {j, k}, but LHS expects {i}", "First", "Second"],
    ),
    (
        "missing_axis_in_matrix_product",
        """
        Out[i,j] = Left[i,k] Right[k,l]
        """,
        [
            "Out: RHS joins on {i, k, l}, projects {k, l}, but LHS expects {i, j}",
            "missing indices j",
        ],
    ),
    (
        "stray_axis_with_index_function",
        """
        Out[i] = In[i] Even(j)
        """,
        [
            "Out: RHS joins on {i, j}, projects {j}, but LHS expects {i}",
            "unmatched RHS indices j from even",
        ],
    ),
    (
        "missing_axis_with_index_function",
        """
        Out[i] = Even(j)
        """,
        ["Out: RHS joins on {j}, projects {j}, but LHS expects {i}", "missing indices i"],
    ),
    (
        "dotted_axis_missing",
        """
        Soft[i.] = In[j]
        """,
        ["Soft: RHS joins on {j}, projects {j}, but LHS expects {i′}", "missing indices i′"],
    ),
    (
        "dotted_axis_with_stray",
        """
        Soft[i.] = In[i] Mask[j]
        """,
        ["Soft: RHS joins on {i′, j}, projects {j}, but LHS expects {i′}", "Mask"],
    ),
    (
        "streaming_axis_missing",
        """
        Hidden[i,*t+1] = Hidden[i]
        """,
        ["Hidden: RHS joins on {i}, projects {∅}, but LHS expects {i, t}", "missing indices t"],
    ),
    (
        "streaming_axis_stray",
        """
        Hidden[i,*t+1] = Hidden[i,*t] Extra[k]
        """,
        [
            "Hidden: RHS joins on {i, k}, projects {k}, but LHS expects {i, t}",
            "missing indices t",
            "Extra",
        ],
    ),
    (
        "slice_axis_missing_from_lhs",
        """
        Out[i] = Block[i,0:2]
        """,
        ["Out: RHS joins on {0:2, i}, projects {0:2}, but LHS expects {i}", "Block"],
    ),
    (
        "slice_axis_missing_on_rhs",
        """
        Out[i,0:2] = Block[i]
        """,
        ["Out: RHS joins on {i}, projects {∅}, but LHS expects {0:2, i}", "missing indices 0:2"],
    ),
    (
        "misnamed_join_axis",
        """
        Out[i] = Left[i] Right[J]
        """,
        ["Out: RHS joins on {J, i}, projects {J}, but LHS expects {i}", "Right"],
    ),
    (
        "stray_axis_in_three_factor_product",
        """
        Out[i,j] = A[i,j] B[j,k] C[p]
        """,
        ["Out: RHS joins on {i, j, k, p}, projects {k, p}, but LHS expects {i, j}", "p from C"],
    ),
    (
        "missing_axis_in_chain",
        """
        Out[i,j] = First[i,j] Second[j,k] Third[k,l]
        """,
        [
            "Out: RHS joins on {i, j, k, l}, projects {k, l}, but LHS expects {i, j}",
            "unmatched RHS indices l from Third",
        ],
    ),
]


@pytest.mark.parametrize(
    "case_name, program_src, expected_fragments",
    SHAPE_ERROR_CASES,
    ids=[case[0] for case in SHAPE_ERROR_CASES],
)
def test_shape_mistake_messages(case_name, program_src, expected_fragments):
    src = textwrap.dedent(program_src).strip()
    with pytest.raises(ValueError) as err:
        Program(src)
    message = str(err.value)
    assert "RHS joins on" in message
    assert "Traceback" not in message
    for fragment in expected_fragments:
        assert fragment in message
