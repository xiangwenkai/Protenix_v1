#!/usr/bin/env python3
"""
Runnable diversity sampling script for Protenix.
Input: input.cif
Output: sampled structures with diversity bias
"""

import torch
import json
from pathlib import Path
from protenix.model.diversity_sampler import DiversitySampler
from protenix.model.generator import sample_diffusion, InferenceNoiseScheduler
from protenix.model.protenix import Protenix
from protenix.data.data_pipeline import DataPipeline


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load Protenix model from checkpoint."""
    model = Protenix.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def prepare_input(cif_path: str, device: str = "cuda"):
    """Prepare input features from CIF file."""
    pipeline = DataPipeline()
    input_features = pipeline.process_cif(cif_path)

    # Move to device
    for key in input_features:
        if isinstance(input_features[key], torch.Tensor):
            input_features[key] = input_features[key].to(device)

    return input_features


def run_sampling(
    model,
    input_features,
    num_rounds: int = 3,
    samples_per_round: int = 4,
    device: str = "cuda",
):
    """Run multi-round diversity sampling."""

    # Initialize diversity sampler
    diversity_sampler = DiversitySampler(
        weight=1.0,
        sigma=2.0,
        n_smooth=1,
        bias_tmin=0.0,
    )

    # Setup noise schedule
    noise_scheduler = InferenceNoiseScheduler()
    noise_schedule = noise_scheduler(N_step=200, device=device)

    # Extract embeddings from model
    with torch.no_grad():
        embeddings = model.trunk(input_features)
        s_inputs = embeddings["s_inputs"]
        s_trunk = embeddings["s_trunk"]
        z_trunk = embeddings["z_trunk"]
        pair_z = embeddings.get("pair_z", None)
        p_lm = embeddings.get("p_lm", torch.zeros_like(s_trunk))
        c_l = embeddings.get("c_l", torch.zeros_like(s_trunk))

    all_samples = []

    for round_idx in range(num_rounds):
        print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")

        # Sample conformations
        with torch.no_grad():
            x_samples = sample_diffusion(
                denoise_net=model.diffusion_module,
                input_feature_dict=input_features,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                noise_schedule=noise_schedule,
                N_sample=samples_per_round,
                diversity_sampler=diversity_sampler,
            )

        # Process samples
        for i in range(samples_per_round):
            structure = x_samples[..., i, :, :]
            all_samples.append(structure.cpu())
            print(f"  Sample {i + 1}: shape={structure.shape}")

        print(f"  Structures in bank: {len(diversity_sampler.structure_bank)}")

    return torch.stack(all_samples, dim=0)


def save_samples(samples: torch.Tensor, output_dir: str = "outputs"):
    """Save sampled structures."""
    Path(output_dir).mkdir(exist_ok=True)

    for i, sample in enumerate(samples):
        output_path = Path(output_dir) / f"sample_{i:03d}.pt"
        torch.save(sample, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.cif", help="Input CIF file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--rounds", type=int, default=3, help="Number of sampling rounds")
    parser.add_argument("--samples", type=int, default=4, help="Samples per round")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    print(f"Preparing input from {args.input}...")
    input_features = prepare_input(args.input, args.device)

    print("Running diversity sampling...")
    samples = run_sampling(
        model,
        input_features,
        num_rounds=args.rounds,
        samples_per_round=args.samples,
        device=args.device,
    )

    print(f"\nSaving {len(samples)} samples...")
    save_samples(samples, args.output)

    print(f"\nDone! Total samples: {len(samples)}")
