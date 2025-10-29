class CosineWithWarmup:
    """Cosine annealing from start->end with warmup in [0, warmup_frac]."""
    def __init__(self, start: float, end: float, warmup_frac: float = 0.1):
        self.start, self.end, self.warmup = float(start), float(end), float(warmup_frac)

    def at(self, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        if t < self.warmup:
            return self.start + (self.end - self.start) * (t / max(self.warmup, 1e-8))
        # cosine on the remaining segment
        import math
        tc = (t - self.warmup) / max(1.0 - self.warmup, 1e-8)
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * tc))
