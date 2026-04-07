#!/usr/bin/env python3
"""
Diversity sampling script - uses inference.py approach.
Usage: python run_diversity_sampling.py --input_json input.json --checkpoint_path model.pt --rounds 3 --samples 1
"""

import json
import copy
import logging
import torch
from pathlib import Path
from typing import Any

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.data.utils import save_structure_cif
from protenix.model.protenix import Protenix
from protenix.model.diversity_sampler import DiversitySampler
from protenix.utils.torch_utils import to_device

logger = logging.getLogger(__name__)


class DiversitySamplingRunner:
    """Runner for diversity sampling with bias injection."""

    def __init__(self, checkpoint_path: str, model_name: str = "protenix_base_default_v1.0.0", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build configs exactly like inference.py
        base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        model_specifics = model_configs.get(model_name, {})

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
            return d

        deep_update(base_configs, model_specifics)

        # Set checkpoint path in configs before parse_configs
        base_configs["load_checkpoint_dir"] = str(Path(checkpoint_path).parent)
        base_configs["model_name"] = model_name

        # Use parse_configs like inference.py (without checkpoint_path arg)
        from protenix.config.config import parse_configs
        arg_str = ""  # No additional args needed
        configs = parse_configs(
            configs=base_configs,
            arg_str=arg_str,
            fill_required_with_null=True,
        )

        # Initialize model
        logger.info(f"Initializing model: {model_name}")
        self.model = Protenix(configs)
        self.model = self.model.to(self.device)

        # Load checkpoint weights
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        sample_key = list(checkpoint["model"].keys())[0]
        print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }

        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=False
        )
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    @torch.no_grad()
    def predict(self, data: dict, diversity_sampler=None) -> dict:
        """Run model prediction."""
        data = to_device(data, self.device)
        prediction, _, _ = self.model(
            input_feature_dict=data["input_feature_dict"],
            label_full_dict=None,
            label_dict=None,
            mode="inference",
            diversity_sampler=diversity_sampler,
        )
        return prediction

    def run_diversity_sampling(self, dataloader, num_rounds: int, samples_per_round: int, output_dir: str = "outputs",
                               bias_weight: float = 1.0, bias_sigma: float = 2.0, bias_n_smooth: int = 1, bias_tmin: float = 0.0):
        """Run multi-round diversity sampling.

        Args:
            bias_weight: Strength of biasing potential (higher = more diversity)
            bias_sigma: Width of Gaussian potential (higher = gentler repulsion)
            bias_n_smooth: Number of neighbors for smoothing (higher = smoother)
            bias_tmin: Minimum noise level for bias injection
        """
        diversity_sampler = DiversitySampler(
            weight=bias_weight,
            sigma=bias_sigma,
            n_smooth=bias_n_smooth,
            bias_tmin=bias_tmin,
        )

        all_samples = []
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        for batch_idx, batch_data in enumerate(dataloader):
            logger.info(f"\n=== Processing batch {batch_idx + 1} ===")

            # Extract batch and atom_array
            batch = batch_data[0][0]
            atom_array = batch_data[0][1]
            entity_poly_type = batch.get("entity_poly_type", {})
            pdb_id = batch.get("pdb_id", f"batch_{batch_idx}")

            # Cache batch for reuse across rounds

            best_ranking_score = 0
            for round_idx in range(num_rounds):
                logger.info(f"Round {round_idx + 1}/{num_rounds}")
                batch = copy.deepcopy(batch_data[0][0])
                with torch.no_grad():
                    # Use cached batch to avoid missing keys
                    prediction = self.predict(batch, diversity_sampler=diversity_sampler)

                # Extract samples
                if prediction is not None:
                    current_score = 0.8*prediction['summary_confidence'][0]['iptm'].item() + 0.2*prediction['summary_confidence'][0]['ptm'].item()
                    print(f"current_score: {current_score}")
                    if best_ranking_score == 0:
                        best_ranking_score = current_score
                    elif current_score > best_ranking_score:
                        best_ranking_score = current_score
                    nice_score = current_score > best_ranking_score * 0.95
                    if nice_score:
                        x_samples = prediction["coordinate"]
                        for i in range(min(samples_per_round, x_samples.shape[0])):
                            is_add = diversity_sampler.add_structure(x_samples[i])
                            sample = x_samples[i].cpu()
                            
                            if is_add:
                                all_samples.append(sample)
                                sample_idx = len(all_samples) - 1

                            # Save as CIF if atom_array available
                            if atom_array is not None and is_add:
                                cif_path = output_path / f"sample_{sample_idx:03d}.cif"
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
                            else:
                                logger.info(f"  Sample {i + 1}: {sample.shape}")

                        logger.info(f"  Bank size: {len(diversity_sampler.structure_bank)}")

            logger.info(f"\nSaved {len(all_samples)} samples to {output_path}")


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
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--bias_weight", type=float, default=1.0, help="Bias strength (higher = more diversity)")
    parser.add_argument("--bias_sigma", type=float, default=2.0, help="Gaussian width (higher = gentler)")
    parser.add_argument("--bias_n_smooth", type=int, default=1, help="Smoothing neighbors")
    parser.add_argument("--bias_tmin", type=float, default=0.0, help="Min noise level for bias")

    args = parser.parse_args()

    # Build configs for dataloader
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    model_specifics = model_configs.get(args.model_name, {})

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specifics)
    base_configs["input_json_path"] = args.input_json
    base_configs["model_name"] = args.model_name

    from protenix.config.config import parse_configs
    configs = parse_configs(
        configs=base_configs,
        arg_str="",
        fill_required_with_null=True,
    )

    # Get dataloader
    logger.info(f"Loading data from {args.input_json}")
    dataloader = get_inference_dataloader(configs=configs)

    # Initialize runner
    runner = DiversitySamplingRunner(args.checkpoint_path, args.model_name, args.device)

    # Run sampling
    runner.run_diversity_sampling(
        dataloader=dataloader,
        num_rounds=args.rounds,
        samples_per_round=args.samples,
        output_dir=args.output_dir,
        bias_weight=args.bias_weight,
        bias_sigma=args.bias_sigma,
        bias_n_smooth=args.bias_n_smooth,
        bias_tmin=args.bias_tmin,
    )


if __name__ == "__main__":
    main()
