import pytest
import torch

from llm_numeric_inputs.devices import clear_cuda_memory, print_cuda_memory, resolve_device


def test_resolve_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_returns_torch_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_cuda_when_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_cuda_index_out_of_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="out of range"):
        resolve_device("cuda:1")


def test_resolve_invalid_device_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_device("tpu")


def test_cuda_helpers_do_not_fail_on_cpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    clear_cuda_memory()
    print_cuda_memory()
    output = capsys.readouterr().out
    assert "CUDA is not available" in output
