import torch

from protenix.model.eclip_binding import (
    EclipSignalLoss,
    align_signal_to_prediction,
    binary_auprc,
    compute_distogram_binding_score,
    compute_soft_binding_score,
    masked_pearson,
    normalize_log_signal,
)


def test_soft_binding_score_tracks_distance_and_has_gradients():
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    feat_dict = {
        "atom_to_token_idx": torch.tensor([0, 1, 2]),
        "is_protein": torch.tensor([1, 0, 0]),
        "is_rna": torch.tensor([0, 1, 1]),
    }

    p_bind, rna_tokens = compute_soft_binding_score(
        coords,
        feat_dict,
        cutoff=5.0,
        temperature=0.5,
        softmin_beta=4.0,
    )

    assert rna_tokens.tolist() == [1, 2]
    assert p_bind.shape == (1, 2)
    assert p_bind[0, 0] > 0.99
    assert p_bind[0, 1] < 1e-6

    p_bind.sum().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert coords.grad.abs().sum() > 0


def test_signal_loss_binarizes_signal_and_weights_positives():
    p_bind = torch.tensor([0.8, 0.1, 0.1], requires_grad=True)
    target, mask = align_signal_to_prediction(torch.tensor([0.0, 10.0]), 3, device=p_bind.device)
    loss_fn = EclipSignalLoss(
        profile_weight=1.0,
        positive_weight=5.0,
        point_weight=2.0,
    )

    loss, metrics = loss_fn(p_bind, target, mask)

    assert torch.isfinite(loss)
    assert metrics["profile_bce"] > 0
    assert metrics["point_loss"] > 0
    assert metrics["target_positive_sum"] == 1
    loss.backward()
    assert p_bind.grad is not None
    assert torch.isfinite(p_bind.grad).all()


def test_distogram_binding_score_uses_max_protein_contact():
    contact_probs = torch.tensor(
        [
            [0.0, 0.2, 0.9, 0.1],
            [0.2, 0.0, 0.3, 0.4],
            [0.9, 0.3, 0.0, 0.8],
            [0.1, 0.4, 0.8, 0.0],
        ]
    )
    feat_dict = {
        "atom_to_token_idx": torch.tensor([0, 1, 2, 3]),
        "is_protein": torch.tensor([1, 1, 0, 0]),
        "is_rna": torch.tensor([0, 0, 1, 1]),
    }

    p_bind, rna_tokens = compute_distogram_binding_score(contact_probs, feat_dict)

    assert rna_tokens.tolist() == [2, 3]
    assert torch.allclose(p_bind, torch.tensor([0.9, 0.4]))


def test_point_loss_uses_per_sample_normalized_log_signal():
    p_bind = torch.tensor([0.0, 0.5, 1.0])
    target_signal = torch.tensor([0.0, 3.0, 15.0])
    expected_target = torch.log1p(target_signal) / torch.log1p(target_signal).max()

    assert torch.allclose(normalize_log_signal(target_signal), expected_target)

    loss_fn = EclipSignalLoss(
        profile_weight=0.0,
        positive_weight=5.0,
        point_weight=1.0,
    )
    loss, metrics = loss_fn(p_bind, target_signal)
    expected_loss = torch.nn.functional.smooth_l1_loss(p_bind, expected_target)

    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(metrics["point_loss"], expected_loss)


def test_binary_auprc_has_random_baseline():
    pred = torch.tensor([0.9, 0.8, 0.1, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    mask = torch.ones(4, dtype=torch.bool)

    auprc = binary_auprc(pred, target, mask)

    assert torch.isfinite(auprc)
    assert 0.0 <= auprc <= 1.0
    assert torch.isclose(auprc, torch.tensor((0.5 + 0.5) / 2))


def test_binary_profile_pearson_uses_positive_labels():
    pred = torch.tensor([0.1, 0.7, 0.2, 0.8])
    raw_signal = torch.tensor([0.0, 2.0, 0.0, 10.0])
    mask = torch.ones(4, dtype=torch.bool)
    binary_signal = (raw_signal > 0).float()

    pearson = masked_pearson(pred, binary_signal, mask)

    assert torch.isfinite(pearson)
    assert pearson > 0
