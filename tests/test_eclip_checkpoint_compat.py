import torch

from protenix.model.eclip_binding import EclipSignalLoss
from runner.train_eclip_ppft import build_eclip_ppft_checkpoint


def test_eclip_sidecar_state_is_not_inside_model_state_dict():
    model = torch.nn.Linear(3, 2)
    signal_loss = EclipSignalLoss()

    checkpoint = build_eclip_ppft_checkpoint(
        model=model,
        signal_loss=signal_loss,
        optimizer=None,
        scheduler=None,
        step=7,
    )

    assert "eclip_signal_loss" in checkpoint
    assert all(not key.startswith("eclip_") for key in checkpoint["model"])

    reloaded = torch.nn.Linear(3, 2)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
