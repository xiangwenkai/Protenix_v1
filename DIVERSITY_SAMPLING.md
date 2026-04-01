# Diversity Sampling for Protenix

## Overview

The `DiversitySampler` module enables diverse conformation sampling during diffusion by injecting repulsive bias based on AF3_ReD implementation. Uses Kabsch alignment, Gaussian potential weighting, and token-level smoothing.

## Key Components

### DiversitySampler Class

Located in `protenix/model/diversity_sampler.py`:

- **Structure Bank**: Maintains previously sampled conformations
- **Kabsch Alignment**: SVD-based optimal rotation alignment
- **Gaussian Potential**: `exp(-MSD / (2 * sigma_eff^2))` weighting
- **Token-level Smoothing**: Residue-based neighbor averaging
- **Noise-dependent Weighting**: `sigma_eff = sigma + noise_level`

## Algorithm

```
For each diffusion step:
1. If structure_bank is empty or noise_level < bias_tmin:
   - Use standard AF3 update: x = x + delta_x_af3

2. Otherwise:
   - For each reference structure x_ref in bank:
     * Center both structures
     * Compute optimal rotation via SVD (Kabsch)
     * Align: x_aligned = x_centered @ R^T
     * Compute MSD = mean((x_aligned - x_ref_centered)^2)

   - Compute Gaussian weights: exp(-MSD / (2 * (sigma + noise_level)^2))
   - Smooth differences at token level (n_smooth neighbors)
   - Compute gradient: grad = -weight * sum(exp * diffs_smoothed) / sigma_eff^2
   - Inject: x = x + delta_x_af3 + grad
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight` | 1.0 | Strength of biasing potential |
| `sigma` | 2.0 | Width of Gaussian potential |
| `n_smooth` | 1 | Residue neighbors for smoothing |
| `bias_tmin` | 0.0 | Min noise level for bias application |

## Integration

```python
from protenix.model.diversity_sampler import DiversitySampler

# 默认对所有链加入偏置和平滑
diversity_sampler = DiversitySampler(
    weight=1.0,
    sigma=2.0,
    n_smooth=1,
)

x_samples = sample_diffusion(
    ...,
    N_sample=4,
    diversity_sampler=diversity_sampler,
)
```

## Key Improvements over Initial Implementation

1. **Kabsch Alignment**: Proper SVD-based rotation instead of simple centering
2. **Gaussian Potential**: Smooth weighting function instead of hard thresholding
3. **Noise Scaling**: `sigma_eff = sigma + noise_level` makes bias weaker at high noise
4. **Token-level Smoothing**: Residue-based averaging instead of spatial Gaussian kernel
5. **Time Threshold**: `bias_tmin` prevents bias at early denoising steps

