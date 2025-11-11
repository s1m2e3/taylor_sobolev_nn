# taylor_sobolev_utils.py

import contextlib
import torch
from torch.autograd.functional import jvp as _jvp, jacobian as _jacobian

try:
    # Optional import: if FSDP isn't available in your build, this still works for plain modules
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _HAS_FSDP = True
except Exception:
    FSDP = None
    _HAS_FSDP = False


def _is_fsdp(module) -> bool:
    return _HAS_FSDP and isinstance(module, FSDP)


def _inner_module(module):
    """
    For FSDP modules, return the wrapped nn.Module; otherwise return the module itself.
    """
    return getattr(module, "_fsdp_wrapped_module", module)


@contextlib.contextmanager
def _maybe_summon_full_params(module):
    """
    If module is FSDP, temporarily gather/unflatten params for functional calls.
    Otherwise, a no-op context manager.
    """
    if _is_fsdp(module):
        # writeback=False so we don't accidentally modify/shard new tensors
        with FSDP.summon_full_params(module, recurse=True, writeback=False):
            yield
    else:
        yield


def estimate_gradient(module: torch.nn.Module,
                      x0: torch.Tensor,
                      displacement: torch.Tensor):
    """
    Compute (f(x0), J_x0 · v) where f is the module's forward, using forward-mode AD.
    - Works with plain nn.Module and FSDP-wrapped modules.
    - Differentiates w.r.t. input only (not params).
    """
    # Ensure tangent matches input dtype/device to avoid AMP/dtype mismatches
    v = displacement.to(dtype=x0.dtype, device=x0.device)

    inner = _inner_module(module)

    def f(inp: torch.Tensor):
        # Call the inner model directly (handles both plain and FSDP-wrapped)
        return inner(inp)

    with _maybe_summon_full_params(module):
        # strict=False tolerates non-differentiable bits in closures; create_graph=False saves memory
        y, j = _jvp(f, (x0,), (v,), create_graph=False, strict=False)

    return y, j


def get_jacobian(module: torch.nn.Module, x: torch.Tensor):
    """
    Compute the Jacobian of module(x) w.r.t. x.
    NOTE: Extremely memory-heavy; prefer JVPs in training loops.
    """
    inner = _inner_module(module)

    def f(inp: torch.Tensor):
        return inner(inp)

    with _maybe_summon_full_params(module):
        J = _jacobian(f, x, create_graph=False, strict=False)

    return J
