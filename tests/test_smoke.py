import torch

from mbndt import MBNDT


def test_binary_forward_is_finite_and_single_output_per_row():
    torch.manual_seed(0)
    model = MBNDT(n_features=4, D=2, B=3, use_masks=True)
    inputs = torch.randn(8, 4)

    logits, auxiliary = model(inputs, return_aux=True)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()
    assert auxiliary["g_soft"].shape == (8, model.num_internal_nodes, 3)
    torch.testing.assert_close(
        auxiliary["g_soft"].sum(dim=-1),
        torch.ones(8, model.num_internal_nodes),
    )
