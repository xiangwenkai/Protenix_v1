#!/usr/bin/env python3
"""
Minimal diversity sampling script - directly runnable.
Usage: python run_diversity_sampling_minimal.py --checkpoint model.pt --input input.cif
"""

import torch
import argparse
from pathlib import Path


def run_diversity_sampling(checkpoint_path: str, input_cif: str, num_rounds: int = 3, samples_per_round: int = 4):
    """
    Minimal example showing how to use DiversitySampler with Protenix.
    """
    from protenix.model.protenix import Protenix
    from protenix.model.diversity_sampler import DiversitySampler
    from protenix.model.generator import sample_diffusion, InferenceNoiseScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = Protenix.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    # Load input (simplified - assumes input_features dict is available)
    print(f"Loading input: {input_cif}")
    # In practice, you'd load from CIF using your data pipeline
    # For now, this is a placeholder
    input_features = torch.load(input_cif) if input_cif.endswith('.pt') else {}

    if not input_features:
        print("ERROR: Could not load input features")
        return

    # Initialize diversity sampler
    diversity_sampler = DiversitySampler(
        weight=1.0,
        sigma=2.0,
        n_smooth=1,
    )

    # Noise schedule
    noise_scheduler = InferenceNoiseScheduler()
    noise_schedule = noise_scheduler(N_step=200, device=device)

    # Get embeddings
    with torch.no_grad():
        embeddings = model.trunk(input_features)

    all_samples = []

    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")

        with torch.no_grad():
            x_samples = sample_diffusion(
                denoise_net=model.diffusion_module,
                input_feature_dict=input_features,
                s_inputs=embeddings["s_inputs"],
                s_trunk=embeddings["s_trunk"],
                z_trunk=embeddings["z_trunk"],
                pair_z=embeddings.get("pair_z"),
                p_lm=embeddings.get("p_lm", torch.zeros_like(embeddings["s_trunk"])),
                c_l=embeddings.get("c_l", torch.zeros_like(embeddings["s_trunk"])),
                noise_schedule=noise_schedule,
                N_sample=samples_per_round,
                diversity_sampler=diversity_sampler,
            )

        for i in range(samples_per_round):
            all_samples.append(x_samples[..., i, :, :].cpu())
            print(f"  Sample {i + 1}: {x_samples[..., i, :, :].shape}")

        print(f"  Bank size: {len(diversity_sampler.structure_bank)}")

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    for i, sample in enumerate(all_samples):
        torch.save(sample, output_dir / f"sample_{i:03d}.pt")
    print(f"\nSaved {len(all_samples)} samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--input", default="input.pt", help="Input features (pt or cif)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    run_diversity_sampling(args.checkpoint, args.input, args.rounds, args.samples)
