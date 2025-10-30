import os
from pathlib import Path

import pytest

from fuse import Program


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize(
    "program_path",
    sorted((EXAMPLES_DIR).glob("*.fuse"), key=lambda p: p.name),
)
def test_fuse_examples_compile_and_run(program_path: Path):
    # Execute from the examples directory so relative asset paths resolve
    cwd = os.getcwd()
    try:
        os.chdir(EXAMPLES_DIR)
        src = program_path.read_text(encoding="utf-8")
        prog = Program(src)
        runner = prog.compile(backend="numpy")
        outputs = runner()
        assert isinstance(outputs, dict)
    finally:
        os.chdir(cwd)

