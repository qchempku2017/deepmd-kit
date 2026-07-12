# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest options for .pth freeze validation tests."""
    parser.addoption(
        "--ckpt",
        action="store",
        default=None,
        help="Path to a SeZM checkpoint .pt file for freeze tests.",
    )
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Path to a frozen .pth model file for runtime tests.",
    )


@pytest.fixture(scope="package", autouse=True)
def clear_cuda_memory(request):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
