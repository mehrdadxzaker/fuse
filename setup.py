"""Setuptools build hooks for Fuse."""

from __future__ import annotations

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(_bdist_wheel):
    """Mark the generated wheel as platform specific for cibuildwheel."""

    def finalize_options(self) -> None:  # noqa: D401 - inherited summary
        """Flag the wheel as non-pure so it receives platform tags."""

        super().finalize_options()
        self.root_is_pure = False


setup(cmdclass={"bdist_wheel": BinaryDistribution})
