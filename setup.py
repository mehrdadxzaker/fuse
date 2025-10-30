"""Setuptools build hooks for Fuse."""

from __future__ import annotations

from setuptools import setup

# The project ships pure Python modules only, so we intentionally avoid
# overriding ``bdist_wheel``.  Leaving the default command class in place
# allows cibuildwheel to detect the wheel as ``py3-none-any`` and skip the
# ``auditwheel repair`` step, which otherwise fails for pure Python wheels.
setup()
