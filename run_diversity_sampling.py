#!/usr/bin/env python3
"""
Diversity sampling script following runner/inference.py structure.
Saves samples in both PT and CIF formats.
Usage: python run_diversity_sampling.py --input_json input.json --checkpoint_path model.pt --rounds 3 --samples 4
"""

import json
import logging
import torch
from pathlib import Path
from typing import Any, Mapping

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs, parse_sys_args
from protenix.data.utils import save_structure_cif
from protenix.model.protenix import Protenix
from protenix.model.diversity_sampler import DiversitySampler
from protenix.utils.torch_utils import to_device

logger = logging.getLogger(__name__)


class DiversitySamplingRunner:
    """Runner for diversity sampling with bias injection."""

    def __init__(self, configs: Any):
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model with configs
        self.model = Protenix(configs)
        self.model = self.model.to(self.device)

        # Load checkpoint
        checkpoint_path = configs.checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        logger.info(f"Model loaded from {checkpoint_path} on {self.device}")

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

            # Extract atom_array and entity_poly_type from batch
            atom_array = batch_data.get("atom_array")
            entity_poly_type = batch_data.get("entity_poly_type", {})
            pdb_id = batch_data.get("pdb_id", f"batch_{batch_idx}")

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

                        sample_idx = len(all_samples) - 1

                        # Save as PT
                        pt_path = output_dir / f"sample_{sample_idx:03d}.pt"
                        torch.save(sample, pt_path)

                        # Save as CIF if atom_array available
                        if atom_array is not None:
                            cif_path = output_dir / f"sample_{sample_idx:03d}.cif"
                            try:
                                save_structure_cif(
                                    atom_array=atom_array,
                                    pred_coordinate=sample,
                                    output_fpath=str(cif_path),
                                    entity_poly_type=entity_poly_type,
                                    pdb_id=pdb_id,
                                )
                                logger.info(f"  Sample {i + 1}: {sample.shape} -> {cif_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save CIF: {e}")
                                logger.info(f"  Sample {i + 1}: {sample.shape} -> {pt_path}")
                        else:
                            logger.info(f"  Sample {i + 1}: {sample.shape} -> {pt_path}")

                logger.info(f"  Bank size: {len(diversity_sampler.structure_bank)}")

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
    parser.add_argument("--model_name", default="protenix_base_default_v1.0.0", help="Model name")

    args = parser.parse_args()

    # Build minimal configs for inference only
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specifics = model_configs.get(args.model_name, {})

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specifics)

    # Manually set required fields to avoid parse_configs validation
    base_configs["checkpoint_path"] = args.checkpoint_path
    base_configs["input_json_path"] = args.input_json
    base_configs["output_dir"] = args.output_dir

    # Create config object directly
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)

    configs = Config(base_configs)

    logger.info(f"Using model: {args.model_name}")

    # Load data
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

# example
# python run_diversity_sampling.py --input_json examples/casp/input_json/8VVJ.json --checkpoint_path checkpoint/protenix_base_default_v1.0.0.pt --rounds 3 --samples 1 --output_dir examples/diversity