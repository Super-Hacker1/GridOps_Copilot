"""Lazy construction of the FRA 1D CNN used by the AMD training notebook."""

from typing import Any


def build_fra_1d_cnn(
    torch_module: Any,
    in_channels: int = 3,
    num_classes: int = 5,
) -> Any:
    """Build the exact hackathon architecture without importing torch eagerly."""

    nn = torch_module.nn

    class FRA1DCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(128, num_classes)

        def forward(self, inputs: Any) -> Any:
            features = self.net(inputs).squeeze(-1)
            return self.head(features)

    FRA1DCNN.__name__ = "FRA1DCNN"
    return FRA1DCNN()
