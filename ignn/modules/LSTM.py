"""LSTM
"""

# pylint: disable=invalid-name
from torch import nn


class LSTM(nn.Module):
    """LSTM."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x, nfc=True):
        _, (h_n, _) = self.lstm(x)
        if nfc:
            return h_n[-1]
        out = self.fc(h_n[-1])
        return out
