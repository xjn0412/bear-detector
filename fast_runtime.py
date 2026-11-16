"""Math-preserving launch-overhead optimizations; no model/state-dict changes."""
from collections import Counter, defaultdict
import math
import torch


def invalid_outputs(value):
    from precision import floating_tensors
    tensors = [t.detach() for t in floating_tensors(value) if t.numel()]
    if not tensors:
        return False
    groups = defaultdict(list)
    for tensor in tensors:
        groups[tensor.device, tensor.dtype].append(tensor)
    with torch.no_grad():
        # Infinity norm detects exactly the same NaN/Inf set without squaring
        # finite large values. It never writes into autograd outputs.
        for group in groups.values():
            maximums = torch._foreach_norm(group, float('inf'))
            if not bool(torch.isfinite(torch.stack(maximums)).all()):
                return True
    return False


class ForwardSnapshot:
    """Same RNG/buffer rollback, batching small device-to-device copies."""
    def __init__(self, model, device):
        self.device = device
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        groups = defaultdict(list)
        for buffer in model.buffers():
            groups[buffer.device, buffer.dtype].append(buffer)
        self.groups = []
        with torch.no_grad():
            for buffers in groups.values():
                saved = [torch.empty_like(b) for b in buffers]
                torch._foreach_copy_(saved, buffers)
                self.groups.append((buffers, saved))

    @torch.no_grad()
    def restore(self):
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state(self.cuda_rng, self.device)
        for buffers, saved in self.groups:
            torch._foreach_copy_(buffers, saved)


@torch.no_grad()
def ema_update(self, model):
    from engine.misc import dist_utils
    if self.before_start < self.start:
        self.before_start += 1
        return
    source = dist_utils.de_parallel(model)
    first_source = next(source.parameters())
    first_target = next(self.module.parameters())
    fingerprint = (id(source), first_source.data_ptr(), first_target.data_ptr())
    cache = getattr(self, '_bear_foreach_cache', None)
    if cache is None or cache[0] != fingerprint:
        source_state = source.state_dict()
        pairs = [(target, source_state[key].detach()) for key, target in self.module.state_dict().items()
                 if target.is_floating_point()]
        counts = Counter((t.device, t.untyped_storage().data_ptr()) for t, _ in pairs)
        groups, aliases = defaultdict(lambda: ([], [])), []
        for target, value in pairs:
            if counts[target.device, target.untyped_storage().data_ptr()] > 1:
                # Preserve sequential updates of shared/overlapping state entries.
                aliases.append((target, value))
            else:
                destinations, sources = groups[target.device, target.dtype]
                destinations.append(target)
                sources.append(value)
        cache = (fingerprint, list(groups.values()), aliases)
        self._bear_foreach_cache = cache
    self.updates += 1
    decay = self.decay_fn(self.updates)
    for targets, sources in cache[1]:
        # Keep the original three operations and rounding points. Using alpha
        # or lerp could introduce fused multiply-add and alter EMA rounding.
        scaled_sources = torch._foreach_mul(sources, 1 - decay)
        torch._foreach_mul_(targets, decay)
        torch._foreach_add_(targets, scaled_sources)
    for target, source_tensor in cache[2]:
        target *= decay
        target += (1 - decay) * source_tensor


def scaler_guard(scaler):
    from engine.misc import dist_utils
    scale = scaler.get_scale()
    valid = math.isfinite(scale) and scale >= 1e-8
    if not dist_utils.is_dist_available_and_initialized():
        if not valid:
            raise FloatingPointError('Dynamic loss scale collapsed; diagnose without retry')
        return
    device = torch.device('cuda', torch.cuda.current_device())
    # One MIN reduction yields both global minimum and negative maximum.
    pair = torch.tensor([scale, -scale] if valid else [-float('inf')]*2,
                        device=device, dtype=torch.float64)
    torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.MIN)
    if not bool(torch.isfinite(pair).all() & (pair[0] == -pair[1])):
        raise FloatingPointError('Invalid or rank-inconsistent GradScaler scale')


def install():
    import precision
    import runtime
    from engine.optim.ema import ModelEMA
    precision.invalid_outputs = invalid_outputs
    precision.ForwardSnapshot = ForwardSnapshot
    runtime.scaler_guard = scaler_guard
    ModelEMA.update = ema_update
