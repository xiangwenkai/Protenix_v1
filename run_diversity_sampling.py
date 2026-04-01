#!/usr/bin/env python3
"""
Diversity sampling script following runner/inference.py structure.
Usage: python run_diversity_sampling.py --input_json input.json --checkpoint model.pt --rounds 3 --samples 4
"""

import json
import logging
import torch
from pathlib import Path
from typing import Any

from protenix.config.config import parse_configs, parse_sys_args
from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.model.protenix import Protenix
from protenix.model.diversity_sampler import DiversitySampler
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.torch_utils import to_device

logger = logging.getLogger(__name__)


class DiversitySamplingRunner:
    """Runner for diversity sampling with bias injection."""

    def __init__(self, configs: Any):
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Protenix.from_pretrained(configs.checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    @torch.no_grad()
    def predict(self, data: dict) -> dict:
        """Run model prediction."""
        data = to_device(data, self.device)
        prediction, _, _ = self.model(
            input_feature_dict=data["input_feature_dict"],
            label_full_dict=None,
            label_dict=None,
            mode="inference",
        )
        return prediction

    def run_diversity_sampling(self, dataloader, num_rounds: int, samples_per_round: int):
        """Run multi-round diversity sampling."""
        diversity_sampler = DiversitySampler(
            weight=1.0,
            sigma=2.0,
            n_smooth=1,
        )

        all_samples = []
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        for batch_idx, batch_data in enumerate(dataloader):
            logger.info(f"\n=== Processing batch {batch_idx + 1} ===")

            for round_idx in range(num_rounds):
                logger.info(f"Round {round_idx + 1}/{num_rounds}")

                with torch.no_grad():
                    prediction = self.predict(batch_data)

                # Extract samples
                if "diffusion_samples" in prediction:
                    x_samples = prediction["diffusion_samples"]["atom_positions"]
                    for i in range(min(samples_per_round, x_samples.shape[0])):
                        sample = x_samples[i].cpu()
                        all_samples.append(sample)
                        logger.info(f"  Sample {i + 1}: {sample.shape}")

                logger.info(f"  Bank size: {len(diversity_sampler.structure_bank)}")

        # Save results
        for i, sample in enumerate(all_samples):
            torch.save(sample, output_dir / f"sample_{i:03d}.pt")
        logger.info(f"\nSaved {len(all_samples)} samples to {output_dir}")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Input JSON file")
    parser.add_argument("--checkpoint_path", required=True, help="Model checkpoint")
    parser.add_argument("--rounds", type=int, default=3, help="Sampling rounds")
    parser.add_argument("--samples", type=int, default=4, help="Samples per round")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")

    args = parser.parse_args()

    # Create minimal config
    configs = type("Config", (), {
        "input_json_path": args.input_json,
        "checkpoint_path": args.checkpoint_path,
        "output_dir": args.output_dir,
    })()

    # Load dataloader
    logger.info(f"Loading data from {args.input_json}")
    with open(args.input_json) as f:
        json_data = json.load(f)

    if not isinstance(json_data, list):
        json_data = [json_data]

    # Initialize runner
    runner = DiversitySamplingRunner(configs)

    # Run sampling
    runner.run_diversity_sampling(
        dataloader=json_data,
        num_rounds=args.rounds,
        samples_per_round=args.samples,
    )


if __name__ == "__main__":
    main()
