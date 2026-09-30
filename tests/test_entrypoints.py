import importlib

import wrapperNM


def test_entrypoint_modules_import_and_build_parsers():
    for module_name in ["alignmentNN", "alignmentTextN", "alignmentMixedN"]:
        module = importlib.import_module(module_name)
        parser = module.build_parser()
        assert parser is not None


def test_wrappernm_reexports_wrapper():
    assert hasattr(wrapperNM, "NumericLMWrapper")
    assert hasattr(wrapperNM, "load_specific_weights")


def test_mixed_entrypoint_parser_is_active():
    import alignmentMixedN

    parser = alignmentMixedN.build_parser()
    args = parser.parse_args(["--config", "config/config.yaml", "--num_epochs", "1"])
    assert args.config == "config/config.yaml"
    assert args.num_epochs == 1
