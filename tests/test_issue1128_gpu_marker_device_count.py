"""#1128 — the ``gpu`` marker must gate on a usable device, not on the build.

``torch.cuda.is_available()`` answers "was torch built with CUDA and can it
initialise", which stays True even when no device is visible — for example
under ``CUDA_VISIBLE_DEVICES=""``, where ``device_count()`` is 0. The shared
probe in ``tests/conftest.py`` returned exactly that, so a ``gpu``-marked test
proceeded on a box with no visible device and *failed* (walking into CPU-only
bitsandbytes) instead of skipping.

The probe now also requires ``device_count() > 0``. A marker that fails open
(runs when it should skip) is the worse half of the property, so these tests
pin *both* directions by driving the real probe with the two states
monkeypatched. They therefore run anywhere — CI included — regardless of
whether the machine has a card.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests import conftest


def _run_probe() -> bool:
    """Call the shared probe with its per-session cache cleared either side."""
    conftest.cuda_available.cache_clear()
    try:
        return conftest.cuda_available()
    finally:
        conftest.cuda_available.cache_clear()


class TestProbeRequiresADevice:
    def test_build_present_but_no_device_is_not_available(self):
        """is_available() True while device_count() 0 -> probe is False (#1128)."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=0),
        ):
            assert _run_probe() is False

    def test_build_present_with_a_visible_device_is_available(self):
        """The other direction: a visible device keeps the probe True."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            assert _run_probe() is True

    def test_no_build_is_not_available(self):
        """is_available() False short-circuits before device_count()."""
        with patch("torch.cuda.is_available", return_value=False):
            assert _run_probe() is False


def _item(*markers):
    by_name = {m.name: m for m in markers}
    return SimpleNamespace(get_closest_marker=lambda name: by_name.get(name))


class TestTheHookHonoursDeviceCount:
    def test_a_gpu_test_skips_when_the_build_is_present_but_no_device(self):
        """End to end through the hook: the exact #1128 state skips, not fails."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=0),
        ):
            conftest.cuda_available.cache_clear()
            try:
                with pytest.raises(pytest.skip.Exception, match="pytest -m gpu"):
                    conftest.pytest_runtest_setup(_item(pytest.mark.gpu.mark))
            finally:
                conftest.cuda_available.cache_clear()

    def test_a_gpu_test_runs_when_a_device_is_visible(self):
        """A visible device still runs the marked test."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            conftest.cuda_available.cache_clear()
            try:
                conftest.pytest_runtest_setup(_item(pytest.mark.gpu.mark))
            finally:
                conftest.cuda_available.cache_clear()
