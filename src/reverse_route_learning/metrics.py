import torch
import torch.nn.functional as F


def js_divergence_from_logits(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between categorical distributions parameterized by logits."""
    lp = F.log_softmax(z1, dim=-1)
    lq = F.log_softmax(z2, dim=-1)
    p, q = lp.exp(), lq.exp()
    m = 0.5 * (p + q)
    lm = torch.log(m.clamp_min(1e-30))
    return 0.5 * ((p * (lp - lm)).sum(-1) + (q * (lq - lm)).sum(-1))


def adaptive_js_drop(curve: list[float]) -> tuple[float, int | None]:
    """Largest early-to-late decrease in a JS-distance curve."""
    if len(curve) < 3:
        return 0.0, None
    best, depth = float("-inf"), None
    h = len(curve)
    for d in range(2, h):
        early = sum(curve[: d - 1]) / (d - 1)
        tail = curve[d - 1 : min(h, d + 1)]
        late = sum(tail) / len(tail)
        drop = early - late
        if drop > best:
            best, depth = drop, d
    return best, depth
