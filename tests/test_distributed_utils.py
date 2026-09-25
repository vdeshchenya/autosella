"""Tests for distributed_validate.utils — pure Python helpers."""

from __future__ import annotations

import argparse
import logging
import re

import pytest

from distributed_validate.worker import get_time, setup_logging, str_to_bool


# ── str_to_bool ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["true", "True", "TRUE", "yes", "YES", "1"])
def test_str_to_bool_truthy(value):
    assert str_to_bool(value) is True


@pytest.mark.parametrize("value", ["false", "False", "FALSE", "no", "NO", "0"])
def test_str_to_bool_falsy(value):
    assert str_to_bool(value) is False


@pytest.mark.parametrize("value", ["maybe", "", "2", "truee", "on", "off"])
def test_str_to_bool_invalid_raises(value):
    with pytest.raises(argparse.ArgumentTypeError, match="Boolean"):
        str_to_bool(value)


# ── setup_logging ────────────────────────────────────────────────────────────

def test_setup_logging_creates_logger_with_file(tmp_path):
    name = "test_component_abc"
    log_dir = tmp_path / "logs"
    logger = setup_logging(name, verbose=False, log_dir=str(log_dir))

    try:
        assert isinstance(logger, logging.Logger)
        assert logger.name == name
        # Log file should be created with component name
        log_file = log_dir / f"{name}.log"
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()
        assert log_file.exists()
        assert "hello" in log_file.read_text()
    finally:
        # Clean up handlers to release file lock
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_setup_logging_verbose_sets_debug_level(tmp_path):
    logger = setup_logging("verbose_component", verbose=True, log_dir=str(tmp_path))
    try:
        # Handler's effective level should be DEBUG
        assert any(h.level == logging.DEBUG for h in logger.handlers)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_setup_logging_no_propagation(tmp_path):
    """Propagation is disabled so messages don't double-print."""
    logger = setup_logging("noprop_component", log_dir=str(tmp_path))
    try:
        assert logger.propagate is False
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_setup_logging_clears_existing_handlers(tmp_path):
    """Calling setup_logging a second time replaces (doesn't accumulate) handlers."""
    logger = setup_logging("reinit_component", log_dir=str(tmp_path))
    n1 = len(logger.handlers)
    logger = setup_logging("reinit_component", log_dir=str(tmp_path))
    n2 = len(logger.handlers)
    try:
        assert n1 == n2  # no accumulation
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def test_setup_logging_accepts_explicit_log_file(tmp_path):
    explicit_file = tmp_path / "custom.log"
    logger = setup_logging(
        "explicit_component", log_dir=str(tmp_path), log_file=str(explicit_file)
    )
    try:
        logger.info("explicit")
        for handler in logger.handlers:
            handler.flush()
        assert explicit_file.exists()
        assert "explicit" in explicit_file.read_text()
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


# ── get_time ─────────────────────────────────────────────────────────────────

def test_get_time_format():
    s = get_time()
    assert isinstance(s, str)
    # Expected: HH:MM:SS
    assert re.match(r"^\d{2}:\d{2}:\d{2}$", s)
