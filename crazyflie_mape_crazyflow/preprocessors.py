"""Custom SKRL preprocessors."""

from __future__ import annotations

import gymnasium
import torch
import torch.nn as nn

from skrl import config
from skrl.utils.spaces.torch import compute_space_size


class PartialRunningStandardScaler(nn.Module):
    """Running standard scaler that only normalizes a subset of input dimensions.

    Dimensions listed in ``skip_dims`` are passed through unchanged, while all
    other dimensions are standardized using Welford's parallel algorithm (same
    as ``RunningStandardScaler``).

    This is useful when an observation vector mixes continuous features
    (positions, velocities) with discrete/binary features (one-hot encodings,
    alive flags) that should not be normalized.
    """

    def __init__(
        self,
        size: int | list[int] | gymnasium.Space,
        *,
        skip_dims: list[int] | None = None,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        """
        :param size: Total size of the input space.
        :param skip_dims: Indices of dimensions to skip (pass through unchanged).
                          If None or empty, behaves identically to RunningStandardScaler.
        :param epsilon: Small number to avoid division by zero.
        :param clip_threshold: Threshold to clip the standardized data.
        :param device: Device for buffers and computation.
        """
        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        self.device = config.torch.parse_device(device)

        total_size = compute_space_size(size, occupied_size=True)
        skip_dims = skip_dims or []

        # Compute which dims to scale vs skip
        all_dims = set(range(total_size))
        skip_set = set(skip_dims)
        scale_dims = sorted(all_dims - skip_set)

        # Store as buffers so they survive .to(device) and state_dict
        self.register_buffer(
            "_scale_indices",
            torch.tensor(scale_dims, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "_skip_indices",
            torch.tensor(sorted(skip_set), dtype=torch.long, device=self.device),
        )

        scale_size = len(scale_dims)
        self._total_size = total_size
        self._has_skip = len(skip_set) > 0

        # Running stats only for the scaled dimensions
        self.register_buffer(
            "running_mean",
            torch.zeros(scale_size, dtype=torch.float64, device=self.device),
        )
        self.register_buffer(
            "running_variance",
            torch.ones(scale_size, dtype=torch.float64, device=self.device),
        )
        self.register_buffer(
            "current_count",
            torch.ones((), dtype=torch.float64, device=self.device),
        )

    def _parallel_variance(
        self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int
    ) -> None:
        """Update running stats using parallel variance algorithm."""
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count
        M2 = (
            (self.running_variance * self.current_count)
            + (input_var * input_count)
            + delta**2 * self.current_count * input_count / total_count
        )
        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = M2 / total_count
        self.current_count = total_count

    def _compute(
        self, x: torch.Tensor, *, train: bool = False, inverse: bool = False
    ) -> torch.Tensor:
        if not self._has_skip:
            # Fast path: scale everything (same as RunningStandardScaler)
            if train:
                if x.dim() == 3:
                    self._parallel_variance(
                        torch.mean(x, dim=(0, 1)),
                        torch.var(x, dim=(0, 1)),
                        x.shape[0] * x.shape[1],
                    )
                else:
                    self._parallel_variance(
                        torch.mean(x, dim=0), torch.var(x, dim=0), x.shape[0]
                    )
            if inverse:
                return (
                    torch.sqrt(self.running_variance.float())
                    * torch.clamp(x, min=-self.clip_threshold, max=self.clip_threshold)
                    + self.running_mean.float()
                )
            return torch.clamp(
                (x - self.running_mean.float())
                / (torch.sqrt(self.running_variance.float()) + self.epsilon),
                min=-self.clip_threshold,
                max=self.clip_threshold,
            )

        # Extract the dims to scale
        x_scale = x[..., self._scale_indices]

        if train:
            if x_scale.dim() == 3:
                self._parallel_variance(
                    torch.mean(x_scale, dim=(0, 1)),
                    torch.var(x_scale, dim=(0, 1)),
                    x_scale.shape[0] * x_scale.shape[1],
                )
            else:
                self._parallel_variance(
                    torch.mean(x_scale, dim=0),
                    torch.var(x_scale, dim=0),
                    x_scale.shape[0],
                )

        if inverse:
            x_scaled = (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(
                    x_scale, min=-self.clip_threshold, max=self.clip_threshold
                )
                + self.running_mean.float()
            )
        else:
            x_scaled = torch.clamp(
                (x_scale - self.running_mean.float())
                / (torch.sqrt(self.running_variance.float()) + self.epsilon),
                min=-self.clip_threshold,
                max=self.clip_threshold,
            )

        # Recombine: put scaled dims back, keep skipped dims as-is
        out = x.clone()
        out[..., self._scale_indices] = x_scaled
        return out

    def forward(
        self,
        x: torch.Tensor | None,
        *,
        train: bool = False,
        inverse: bool = False,
        no_grad: bool = True,
    ) -> torch.Tensor | None:
        if x is None:
            return None
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse)
        return self._compute(x, train=train, inverse=inverse)
