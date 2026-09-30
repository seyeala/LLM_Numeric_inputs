import importlib
import sys

import pytest

import wrapperNM


def test_entrypoint_modules_import_and_build_parsers():
    for module_name in ["alignmentNN", "alignmentTextN", "alignmentMixedN"]:
        module = importlib.import_module(module_name)
        parser = module.build_parser()
        assert parser is not None


def test_wrappernm_reexports_wrapper():
    assert hasattr(wrapperNM, "NumericLMWrapper")
    assert hasattr(wrapperNM, "load_specific_weights")


def test_mixed_entrypoint_reports_cp8_deferral(monkeypatch):
    import alignmentMixedN

    monkeypatch.setattr(sys, "argv", ["alignmentMixedN.py", "--config", "config/config.yaml"])
    with pytest.raises(SystemExit, match="disabled in CP3"):
        alignmentMixedN.main()
