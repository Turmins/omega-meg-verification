import random
import numpy as np
import torch
from omega_dl.seeding import seed_everything

def test_seed_resets_numpy_torch_and_python_generators():
    seed_everything(1337)
    first = (np.random.normal(size=8), torch.randn(8), random.random())
    seed_everything(1337)
    second = (np.random.normal(size=8), torch.randn(8), random.random())
    np.testing.assert_array_equal(first[0], second[0])
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    assert first[2] == second[2]
