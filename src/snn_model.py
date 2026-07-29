import os
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Tuple
from src.utils import get_logger, ensure_dirs

logger = get_logger()


# Model Definition
class EEGDepression_SNN(nn.Module):
    # Three-layer SNN:
    #   FC1 -> LIF1 -> FC2 -> LIF2 -> FC3 (output)

    # LIF parameters:
    #   beta  : membrane potential decay factor (0 < beta < 1)
    #   spike_grad: surrogate gradient for backpropagation through spikes

    def __init__(
        self,
        n_inputs:  int = 140,
        n_hidden:  int = 128,
        n_outputs: int = 2,
        beta:      float = 0.9
    ) -> None:
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Linear layers
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
        self.fc3 = nn.Linear(n_hidden // 2, n_outputs)

        # LIF neuron layers
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass over T time steps.

        # Parameters
        # x : (T, batch, n_inputs)  — spike-encoded input

        # Returns
        # spk_rec : (T, batch, n_outputs)  spike output at each time step
        # mem_rec : (T, batch, n_outputs)  membrane potential at each step
    
        T = x.shape[0]
        spk_rec = []
        mem_rec = []

        # Reset hidden states at start of each new sequence
        self.lif1.reset_mem()
        self.lif2.reset_mem()

        # Create a temporary output LIF to track final layer membrane
        mem_out = torch.zeros(x.shape[1], self.fc3.out_features, device=x.device)

        for t in range(T):
            # Layer 1
            cur1 = self.fc1(x[t])
            spk1 = self.lif1(cur1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2 = self.lif2(cur2)

            # Output layer (integrate but don't spike - use for classification)
            cur3 = self.fc3(spk2)
            mem_out = 0.9 * mem_out + cur3   # leaky integrator

            spk_rec.append(spk2)             # record hidden spikes
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# Prediction Helper
def predict_snn(
    model: EEGDepression_SNN,
    spikes: torch.Tensor,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()
    spikes = spikes.to(device)

    with torch.no_grad():
        _, mem_rec = model(spikes)           # (T, batch, n_outputs)
    logits_sum = mem_rec.sum(dim=0)          # sum over time -> (batch, n_outputs)
    preds      = logits_sum.argmax(dim=1)    # (batch,)

    return preds, logits_sum


# Persistence
def save_snn(model: EEGDepression_SNN, filepath: str = "models/snn_model.pt") -> None:
    ensure_dirs(os.path.dirname(filepath))
    torch.save(model.state_dict(), filepath)
    logger.info(f"SNN weights saved -> {filepath}")


def load_snn(
    filepath: str,
    n_inputs:  int = 140,
    n_hidden:  int = 128,
    n_outputs: int = 2,
    beta:      float = 0.9,
    device:    str = "cpu"
) -> EEGDepression_SNN:
    model = EEGDepression_SNN(n_inputs=n_inputs, n_hidden=n_hidden,
                               n_outputs=n_outputs, beta=beta)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"SNN loaded <- {filepath}")
    return model
