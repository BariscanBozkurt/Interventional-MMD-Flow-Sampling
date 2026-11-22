"""Cached reference-population fields for independent RFF-flow requests.

The reference cloud records a_t = target_mean - mean(psi(reference_t)) before
each Euler update. New particles follow J_psi(x).T @ a_t without estimating a
source mean from their requested batch. Conditional on a frozen cache, this is
one deterministic map, independent of how query particles are grouped.

This removes requested-batch interactions; it does not establish agreement
between the transported law and the interventional target. The reference
initializer, target, frequencies, bandwidth, step size and time grid must stay
fixed. The experiment runner is responsible for saving these cache identities.
"""

import math

import torch

from rff_imq import _feature_inputs, _flow_arguments, _same_type


def _target_inputs(initial, target_mean, omega, c, step_size, n_iterations):
    _flow_arguments(initial, step_size, n_iterations)
    _feature_inputs(initial, omega, c)
    _same_type(initial, target_mean)
    if target_mean.shape != (2 * len(omega),):
        raise ValueError("target_mean must contain 2*len(omega) coordinates")


def _field_inputs(initial, coefficients, omega, c, step_size, stride):
    _flow_arguments(initial, step_size, 0)
    _feature_inputs(initial, omega, c)
    _same_type(initial, coefficients)
    if coefficients.ndim != 2 or coefficients.shape[1] != 2 * len(omega):
        raise ValueError("coefficients must have shape (n_steps, 2*len(omega))")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")


def _velocity(points, coefficient, omega, scale):
    phase = points @ omega.T
    cosine, sine = phase.cos(), phase.sin()
    return scale * (-sine * coefficient[:len(omega)]
                    + cosine * coefficient[len(omega):]) @ omega


@torch.no_grad()
def reference_velocity(points, coefficient, omega, c):
    """Evaluate the prescribed velocity J_psi(points).T @ coefficient."""
    _feature_inputs(points, omega, c)
    _same_type(points, coefficient)
    if coefficient.shape != (2 * len(omega),):
        raise ValueError("coefficient must contain 2*len(omega) coordinates")
    return _velocity(points, coefficient, omega, 1 / math.sqrt(c * len(omega)))


@torch.no_grad()
def record_reference_flow(initial, target_mean, omega, c, step_size, n_iterations):
    """Return reference endpoints and coefficients recorded before each update.

    Parameters are the same as ``rff_imq.rff_euler_flow``. The coefficient
    tensor has shape (n_iterations, 2*len(omega)), remains on the input device,
    and uses the input precision. No normalization of the target is performed.
    The zero-iteration case returns an empty coefficient tensor and a copy of
    the initial particles. Inputs are not changed.
    """
    _target_inputs(initial, target_mean, omega, c, step_size, n_iterations)
    particles = initial.detach().clone()
    coefficients = initial.new_empty((n_iterations, 2 * len(omega)))
    scale = 1 / math.sqrt(c * len(omega))
    for t in range(n_iterations):
        phase = particles @ omega.T
        cosine, sine = phase.cos(), phase.sin()
        coefficients[t, :len(omega)] = target_mean[:len(omega)] - scale * cosine.mean(0)
        coefficients[t, len(omega):] = target_mean[len(omega):] - scale * sine.mean(0)
        velocity = scale * (-sine * coefficients[t, :len(omega)]
                            + cosine * coefficients[t, len(omega):]) @ omega
        particles.add_(velocity, alpha=step_size)
    return particles, coefficients


@torch.no_grad()
def integrate_coefficient_blocks(coefficients, stride=1):
    """Sum successive time coefficients, including the last incomplete block.

    Applying each block sum at one particle state is a coarse Euler
    approximation: it integrates the prescribed coefficient sequence exactly
    in time while freezing J_psi(x) within the block. It is not the original
    sequence of fine updates when stride > 1. The caller still uses the
    original fine step size, not stride times that step size.
    """
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a matrix")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if stride == 1 or len(coefficients) == 0:
        return coefficients
    return torch.stack([coefficients[start:start + stride].sum(0)
                        for start in range(0, len(coefficients), stride)])


def _apply_into(particles, initial, coefficients, omega, scale, step_size):
    """Fixed-shape operations, also suitable for CUDA graph capture."""
    particles.copy_(initial)
    for coefficient in coefficients:
        particles.add_(_velocity(particles, coefficient, omega, scale), alpha=step_size)
    return particles


@torch.no_grad()
def apply_reference_flow(initial, coefficients, omega, c, step_size, stride=1):
    """Apply one frozen field sequence to any nonempty batch of fresh particles.

    ``stride=1`` replays the full Euler time grid. Larger strides perform one
    update using the sum of coefficients in each consecutive block, including
    a short final block. This optional acceleration changes the discretization
    and must be assessed separately. Inputs remain unchanged.
    """
    _field_inputs(initial, coefficients, omega, c, step_size, stride)
    integrated = integrate_coefficient_blocks(coefficients, stride)
    particles = torch.empty_like(initial)
    return _apply_into(particles, initial, integrated, omega,
                       1 / math.sqrt(c * len(omega)), step_size)


@torch.no_grad()
def independent_batch_rff_flow(initial, target_mean, omega, c,
                               step_size, n_iterations):
    """Run independent interacting request clouds in one vectorized call.

    ``initial`` has shape (n_groups, request_size, latent_dim). Each group uses
    its own source feature mean. Groups do not interact. All groups must have
    the same actual request size: do not pad a remainder group with particles.
    This routine supports distributional evaluation efficiently; its joint
    runtime is not the latency of a standalone request.
    """
    if initial.ndim != 3 or any(size < 1 for size in initial.shape):
        raise ValueError("initial must have nonempty (groups, batch, dimension) shape")
    _target_inputs(initial.reshape(-1, initial.shape[-1]), target_mean,
                   omega, c, step_size, n_iterations)
    particles = initial.detach().clone()
    scale = 1 / math.sqrt(c * len(omega))
    for _ in range(n_iterations):
        phase = particles @ omega.T
        cosine, sine = phase.cos(), phase.sin()
        residual_cos = scale * cosine.mean(1, keepdim=True) - target_mean[:len(omega)]
        residual_sin = scale * sine.mean(1, keepdim=True) - target_mean[len(omega):]
        gradient = scale * (-sine * residual_cos + cosine * residual_sin) @ omega
        particles.add_(gradient, alpha=-step_size)
    return particles


class CUDAGraphReferenceSampler:
    """Fixed-shape CUDA-graph replay of the same prescribed Euler map.

    Construction includes coefficient-block integration, warmup and graph
    capture. Measure that setup separately from request latency. A call copies
    a new device-resident initializer into a fixed buffer, replays the graph,
    and by default copies the output into a fresh tensor. It does not sample
    initial ancestors or decode images. With ``clone_output=False``, the output
    aliases an internal buffer and is overwritten by the next call.

    Frequencies and coefficients are cloned during setup to freeze their
    identity. A sampler supports only one batch shape, device and dtype. It
    is not safe for concurrent calls from different streams or host threads.
    """

    @torch.no_grad()
    def __init__(self, initial, coefficients, omega, c, step_size,
                 stride=1, warmup=2):
        _field_inputs(initial, coefficients, omega, c, step_size, stride)
        if initial.device.type != "cuda":
            raise ValueError("CUDA graph sampling requires CUDA tensors")
        if not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        self.input_buffer = initial.detach().clone()
        self.output_buffer = torch.empty_like(initial)
        self.omega = omega.detach().clone()
        self.coefficients = integrate_coefficient_blocks(coefficients, stride).detach().clone()
        self.scale = 1 / math.sqrt(c * len(omega))
        self.step_size = step_size
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(initial.device):
            stream = torch.cuda.Stream(device=initial.device)
            stream.wait_stream(torch.cuda.current_stream(initial.device))
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self._execute()
            torch.cuda.current_stream(initial.device).wait_stream(stream)
            with torch.cuda.graph(self.graph, stream=stream):
                self._execute()
            torch.cuda.current_stream(initial.device).wait_stream(stream)

    def _execute(self):
        return _apply_into(self.output_buffer, self.input_buffer, self.coefficients,
                           self.omega, self.scale, self.step_size)

    @torch.no_grad()
    def __call__(self, initial, clone_output=True):
        if (initial.shape != self.input_buffer.shape
                or initial.device != self.input_buffer.device
                or initial.dtype != self.input_buffer.dtype):
            raise ValueError("query shape, device and dtype must match the captured graph")
        self.input_buffer.copy_(initial)
        self.graph.replay()
        return self.output_buffer.clone() if clone_output else self.output_buffer
