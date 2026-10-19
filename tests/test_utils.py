import torch

from mini_llm.utils import get_device, set_seed


def test_get_device_returns_usable_device():
    device = get_device()
    assert device in {"cuda", "mps", "cpu"}
    x = torch.ones(2, 3, device=device)  # (2, 3)
    assert (x + x).sum().item() == 12.0


def test_set_seed_is_reproducible():
    set_seed(0)
    a = torch.randn(4)
    set_seed(0)
    b = torch.randn(4)
    assert torch.equal(a, b)
