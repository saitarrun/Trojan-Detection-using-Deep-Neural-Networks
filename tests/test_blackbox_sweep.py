import torch
from torch.utils.data import DataLoader, TensorDataset

from blackbox_sweep import black_box_trigger_sweep


class SquareTriggerModel(torch.nn.Module):
    def forward(self, inputs):
        logits = torch.zeros(inputs.size(0), 4, device=inputs.device)
        logits[:, 0] = 1.0
        corner = inputs[:, :, -4:, -4:].mean(dim=(1, 2, 3))
        triggered = corner > 0.9
        logits[triggered, 0] = 0.0
        logits[triggered, 2] = 8.0
        return logits


def test_black_box_trigger_sweep_detects_prediction_collapse():
    images = torch.zeros(32, 3, 32, 32)
    labels = torch.zeros(32, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    result = black_box_trigger_sweep(
        SquareTriggerModel(),
        loader,
        device=torch.device("cpu"),
        num_batches=2,
        trigger_types=["checkerboard", "square"],
    )

    assert result["blackbox_sweep_target"] == 2
    assert result["blackbox_sweep_trigger"] == "square"
    assert result["blackbox_sweep_lift"] == 1.0
    assert result["blackbox_sweep_risk"] > 0.8
