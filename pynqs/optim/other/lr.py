import math


def WSD_Learning_rate(
    step: int,
    W: int = 1000,  # W
    T: int = 3000,  # T
    S: int = 1000,  # S
    lr_max: float = 5e-4,  # eta
    lr_min: float = 1e-4,
    p: int = 3,
    k: float = -1e-3,
):
    if step < W:
        # Warmup
        ratio = step / W
        return lr_max * (ratio**p)
    elif step < W + T:
        # Stable
        return lr_max
    elif step < W + T + S:
        # Decay
        decay_step = step - W - T
        return lr_max * math.exp(k * decay_step)
    else:
        return lr_min
