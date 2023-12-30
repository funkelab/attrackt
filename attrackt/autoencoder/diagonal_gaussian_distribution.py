import math
from typing import Tuple

import torch


class DiagonalGaussianDistribution:
    """
    parameters: (B, 2*C, *spatial)
      first C channels = mean, next C = logvar
    Works for 2D (B,C,H,W), 3D (B,C,D,H,W), etc.
    """

    def __init__(
        self, parameters: torch.Tensor, clamp: Tuple[float, float] = (-30.0, 20.0)
    ):
        self.parameters = parameters
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = logvar.clamp(min=clamp[0], max=clamp[1])

        self.mean = mean
        self.logvar = logvar
        self.var = torch.exp(logvar)
        self.std = torch.exp(0.5 * logvar)

    def _reduce_dims(self, x: torch.Tensor):
        # sum over all non-batch dims
        return tuple(range(1, x.ndim))

    def rsample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            eps = torch.randn_like(self.mean)
            return self.mean + self.std * eps

    def get_mean(self) -> torch.Tensor:
        return self.mean

    def kl(self, reduce: str = "sum"):
        """
        KL(q || N(0, I)).
        reduce: "none" -> elementwise, "sum" -> per-example sum, "mean" -> batch mean
        """
        kl_elems = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
        if reduce == "none":
            return kl_elems
        per_ex = kl_elems.sum(dim=self._reduce_dims(kl_elems))
        if reduce == "sum":
            return per_ex
        if reduce == "mean":
            return per_ex.mean()
        raise ValueError(f"Unknown reduce='{reduce}'")

    def nll(self, z: torch.Tensor, reduce: str = "sum", include_const: bool = True):
        """
        NLL under this Gaussian (i.e., -log q(z|x)); not the VAE recon term.
        """
        const = 0.5 * math.log(2.0 * math.pi) if include_const else 0.0
        nll_elems = const + 0.5 * (self.logvar + (z - self.mean).pow(2) / self.var)
        if reduce == "none":
            return nll_elems
        per_ex = nll_elems.sum(dim=self._reduce_dims(nll_elems))
        if reduce == "sum":
            return per_ex
        if reduce == "mean":
            return per_ex.mean()
        raise ValueError(f"Unknown reduce='{reduce}'")
