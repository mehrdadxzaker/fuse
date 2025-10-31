import os
import sys


def pytest_sessionstart(session):  # noqa: D401 - test harness helper
    """Ensure the current Python's bin directory is on PATH for subprocesses.

    Some environments may lack a global 'python' shim. The CLI smoke test
    launches a subprocess using 'python -m fuse', so we prepend the directory
    containing the running interpreter to PATH to make the 'python' launcher
    discoverable (e.g., .venv/bin/python).
    """

    bin_dir = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if bin_dir and bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([bin_dir, path]) if path else bin_dir

