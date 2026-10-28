"""GDN2 dimensions and exact FLA 0.5.2 parameter counts; no GPU imports."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GDN2Spec:
    d_model: int
    n_heads: int
    head_dim: int = 128
    expand_v: float = 2.0
    n_v_heads: int | None = None
    conv_size: int = 4
    conv_bias: bool = False

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "head_dim", "conv_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.head_dim > 256:
            raise ValueError("FLA GDN2 requires head_dim <= 256")
        if not math.isfinite(self.expand_v) or self.expand_v <= 0:
            raise ValueError("expand_v must be finite and positive")
        expanded = self.head_dim * self.expand_v
        if not math.isclose(expanded, int(expanded), rel_tol=1e-5) or int(expanded) < 1:
            raise ValueError("head_dim * expand_v must be a positive integer")
        if self.n_v_heads is not None:
            if not isinstance(self.n_v_heads, int) or self.n_v_heads < self.n_heads:
                raise ValueError("n_v_heads must be an integer >= n_heads")
            if self.n_v_heads % self.n_heads:
                raise ValueError("n_v_heads must be divisible by n_heads")

    @property
    def value_heads(self) -> int:
        return self.n_heads if self.n_v_heads is None else self.n_v_heads

    @property
    def head_v_dim(self) -> int:
        return int(self.head_dim * self.expand_v)

    @property
    def key_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def value_dim(self) -> int:
        return self.value_heads * self.head_v_dim

    @property
    def num_params(self) -> int:
        d, k, v, r = self.d_model, self.key_dim, self.value_dim, self.head_v_dim
        return (
            d * (2 * k + v)                 # q/k/v
            + r * (d + k)                  # low-rank decay f_proj
            + d * (k + v)                  # erase b_proj, write w_proj
            + self.n_heads + k             # A_log, channel-wise dt_bias
            + r * (d + v) + v              # low-rank output gate, including bias
            + r + v * d                    # output norm and o_proj
            + (self.conv_size + int(self.conv_bias)) * (2 * k + v)
        )

    @property
    def state_elements(self) -> int:
        return self.value_heads * self.head_dim * self.head_v_dim
