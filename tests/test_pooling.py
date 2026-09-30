import pytest
import torch

from llm_numeric_inputs.pooling import last_valid_token_pool


def test_last_valid_token_pool_selects_unpadded_position():
    hidden = torch.tensor([[[1.0], [2.0], [99.0]], [[4.0], [5.0], [6.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = last_valid_token_pool(hidden, mask)
    assert pooled.tolist() == [[2.0], [6.0]]


def test_all_padding_raises():
    hidden = torch.zeros(1, 2, 3)
    mask = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="no valid tokens"):
        last_valid_token_pool(hidden, mask)


@pytest.mark.parametrize(
    "hidden, mask, message",
    [
        (torch.zeros(2, 3), torch.ones(2, 3), "rank 3"),
        (torch.zeros(2, 3, 4), torch.ones(2, 3, 1), "rank 2"),
        (torch.zeros(2, 3, 4), torch.ones(2, 2), "must match"),
    ],
)
def test_invalid_pooling_shapes_raise(hidden, mask, message):
    with pytest.raises(ValueError, match=message):
        last_valid_token_pool(hidden, mask)
