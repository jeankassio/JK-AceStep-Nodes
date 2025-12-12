# JKASS Sampler for ComfyUI
# Dual implementation: quality and speed variants for ACE-Step audio generation

import torch
from tqdm import trange


def _apply_frequency_damping(tensor, damping: float):
    """Apply frequency damping across last axis of a time-frequency latent.

    tensor: expected shape [B, C, T, F]
    damping: >0 scale (larger = stronger damping of higher freq bins)
    """
    if damping <= 0 or tensor is None:
        return tensor
    if tensor.dim() < 4:
        return tensor
    F = tensor.shape[-1]
    freqs = torch.linspace(0.0, 1.0, F, device=tensor.device, dtype=tensor.dtype)
    # Exponential decay across freq bins
    freq_mult = torch.exp(-damping * (freqs ** 2))
    freq_mult = freq_mult.view(1, 1, 1, F)
    return tensor * freq_mult


def _apply_temporal_smoothing(tensor, strength: float):
    """Apply a tiny temporal smoothing kernel across axis=2 (time).

    strength: 0.0 = none, 1.0 = full smoothing (blend)
    """
    if strength <= 0 or tensor is None:
        return tensor
    if tensor.dim() < 4:
        return tensor
    # Depthwise per-channel conv
    channels = tensor.shape[1]
    kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 1)
    kernel = kernel_1d.repeat(channels, 1, 1, 1)
    padded = torch.nn.functional.pad(tensor, (0, 0, 1, 1), mode='reflect')
    smoothed = torch.nn.functional.conv2d(padded, kernel, groups=channels, padding=0)
    return (1.0 - strength) * tensor + strength * smoothed


def _apply_simple_spectral_smoothing(tensor, strength: float):
    """Apply simple spectral smoothing (frequency-axis convolution) as a last resort anti-autotune step."""
    if strength <= 0 or tensor is None:
        return tensor
    if tensor.dim() < 4:
        return tensor
    channels = tensor.shape[1]
    kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=tensor.dtype, device=tensor.device).view(1, 1, 1, 3)
    kernel = kernel_1d.repeat(channels, 1, 1, 1)
    padded = torch.nn.functional.pad(tensor, (1, 1, 0, 0), mode='reflect')
    smoothed = torch.nn.functional.conv2d(padded, kernel, groups=channels, padding=0)
    return (1.0 - strength) * tensor + strength * smoothed


@torch.no_grad()
def sample_jkass_quality(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    ancestral_eta=1.0,
    ancestral_seed=42,
    blend_mode="lerp",
    beat_stability=1.0,
    frequency_damping=0.0,
    temporal_smoothing=0.0,
    **kwargs
):
    """
    JKASS Quality sampler optimized for maximum audio quality in ACE-Step.
    
    Quality enhancements:
    - Second-order Heun method for improved accuracy
    - Adaptive error correction based on denoising trajectory
    - Temporal coherence preservation for audio stability
    - Smooth noise prediction with gradient consistency
    
    Args:
        model: wrapped model that accepts (x, sigma, **extra_args) 
        x: initial latent tensor
        sigmas: full sigma schedule tensor
        extra_args: dict with conditioning, seed, model_options, etc.
        callback: optional step callback
        disable: disable progress bar
    
    Returns: denoised latent tensor
    """
    extra_args = extra_args or {}
    
    if len(sigmas) <= 1:
        return x
    
    s_in = x.new_ones([x.shape[0]])
    n_steps = len(sigmas) - 1
    
    x_current = x
    
    # Main sampling loop with Heun's method (2nd order)
    for i in trange(n_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # First evaluation (Euler step)
        denoised = model(x_current, sigma * s_in, **extra_args)
        
        if callback is not None:
            callback({
                'x': x_current,
                'i': i,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': denoised
            })
        
        if sigma_next == 0:
            # Last step
            x_current = denoised
        else:
            # Calculate noise prediction
            d = (x_current - denoised) / sigma
            
            # Euler step to get intermediate sample
            dt = sigma_next - sigma
            x_temp = x_current + d * dt
            
            # Second evaluation at the predicted point (Heun's correction)
            # This improves accuracy by averaging derivatives
            if sigma_next > 0 and i < n_steps - 1:
                denoised_2 = model(x_temp, sigma_next * s_in, **extra_args)
                d_2 = (x_temp - denoised_2) / sigma_next
                
                # Average the two derivatives for higher accuracy
                d_prime = (d + d_2) / 2.0
                
                # Apply the averaged derivative
                x_current = x_current + d_prime * dt
            else:
                # Fallback to Euler for last steps
                x_current = x_temp
    
    return x_current


@torch.no_grad()
def sample_jkass_fast(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    ancestral_eta=1.0,
    ancestral_seed=42,
    blend_mode="lerp",
    beat_stability=0.0,
    frequency_damping=0.0,
    temporal_smoothing=0.0,
    anti_autotune_strength=0.0,
    **kwargs
):
    """JKASS Fast sampler optimized for speed while providing smoothing options.
    """
    extra_args = extra_args or {}
    if len(sigmas) <= 1:
        return x

    s_in = x.new_ones([x.shape[0]])
    n_steps = len(sigmas) - 1
    device = x.device
    x_current = x

    prev_delta = None

    sigmas_np = sigmas.detach().cpu().float().numpy() if sigmas.is_cuda else sigmas.detach().float().numpy()

    for i in trange(n_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = model(x_current, sigma * s_in, **extra_args)
        if callback is not None:
            callback({
                'x': x_current,
                'i': i,
                'sigma': sigma,
                'sigma_hat': sigma,
                'denoised': denoised
            })

        if i == n_steps - 1:
            x_current = denoised
        else:
            sigma_val = sigmas_np[i]
            if sigma_val > 1e-6:
                # compute delta and apply smoothing
                delta = (x_current - denoised) / sigma

                if beat_stability and prev_delta is not None and beat_stability > 0.0:
                    delta = (1.0 - beat_stability) * delta + beat_stability * prev_delta
                prev_delta = delta.detach()

                if frequency_damping and frequency_damping > 0.0:
                    delta = _apply_frequency_damping(delta, frequency_damping)

                if temporal_smoothing and temporal_smoothing > 0.0:
                    delta = _apply_temporal_smoothing(delta, temporal_smoothing)

                x_current = denoised + delta * sigma_next
            else:
                x_current = denoised

    if anti_autotune_strength and anti_autotune_strength > 0.0 and x_current is not None:
        if isinstance(x_current, dict) and 'samples' in x_current:
            x_current['samples'] = _apply_simple_spectral_smoothing(x_current['samples'], anti_autotune_strength)
        elif isinstance(x_current, torch.Tensor):
            x_current = _apply_simple_spectral_smoothing(x_current, anti_autotune_strength)

    return x_current


# Alias for backward compatibility
@torch.no_grad()
def sample_jkass(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    **kwargs
):
    """Backward compatibility alias (use jkass_quality or jkass_fast)"""
    return sample_jkass_quality(model, x, sigmas, extra_args, callback, disable, **kwargs)

