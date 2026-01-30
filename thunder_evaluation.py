#!/usr/bin/env python3
"""
THUNDER Benchmark Evaluation Script for OpenMidnight

This script allows you to:
1. Evaluate the official OpenMidnight from HuggingFace
2. Evaluate your own trained checkpoint
3. Compare different checkpoints

Usage:
    # Evaluate HuggingFace version
    python thunder_evaluation.py --model openmidnight --dataset mhist --task simple_shot

    # Evaluate your custom checkpoint
    python thunder_evaluation.py --checkpoint /path/to/checkpoint.pth --dataset mhist --task simple_shot

    # Run all key benchmarks
    python thunder_evaluation.py --model openmidnight --all-tasks

    # Compare few-shot vs linear probe
    python thunder_evaluation.py --model openmidnight --dataset mhist --task linear_probing simple_shot knn
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================
DATASETS_QUICK = ["mhist"]
DATASETS_STANDARD = ["mhist", "bach", "break_his", "crc", "patch_camelyon"]
DATASETS_FULL = [
    "mhist", "bach", "break_his", "crc", "patch_camelyon",
    "bracs", "consep", "gleason", "monusac", "panda"
]
DATASETS_SPIDER = ["spider_breast", "spider_colorectal", "spider_skin", "spider_thorax"]

TASKS_ALL = ["linear_probing", "knn", "simple_shot", "image_retrieval"]
TASKS_CLASSIFICATION = ["linear_probing", "knn", "simple_shot"]


# =============================================================================
# MODEL LOADING
# =============================================================================
class OpenMidnightCustom:
    """Wrapper for custom OpenMidnight checkpoint for THUNDER evaluation."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path

        logger.info(f"Loading OpenMidnight from checkpoint: {checkpoint_path}")

        # Load base DINOv2 ViT-G architecture
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "teacher" in checkpoint:
            state_dict = checkpoint["teacher"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove DINO/iBOT head keys if present
        keys_to_remove = [k for k in state_dict.keys() if "dino" in k or "ibot" in k]
        for key in keys_to_remove:
            state_dict.pop(key, None)

        # Map keys to match DINOv2 architecture
        new_state_dict = {}
        model_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(state_dict.keys())

        for ck, mk in zip(checkpoint_keys, model_keys):
            if state_dict[ck].shape == self.model.state_dict()[mk].shape:
                new_state_dict[mk] = state_dict[ck]
            else:
                logger.warning(f"Shape mismatch for {mk}: checkpoint {state_dict[ck].shape} vs model {self.model.state_dict()[mk].shape}")

        # Handle pos_embed shape difference (224 vs 392 resolution)
        if "pos_embed" in state_dict:
            pos_embed = state_dict["pos_embed"]
            self.model.pos_embed = torch.nn.parameter.Parameter(pos_embed)
            new_state_dict["pos_embed"] = pos_embed

        # Load the state dict
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

        self.model = self.model.to(device)
        self.model.eval()

        # Set up transforms (same as official OpenMidnight)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Model metadata for THUNDER
        self.name = "openmidnight_custom"
        self.emb_dim = 1536  # ViT-G embedding dimension
        self.vlm = False

        logger.info("Model loaded successfully")

    def __call__(self, x):
        """Forward pass - returns CLS token."""
        with torch.no_grad():
            return self.model(x)

    def forward(self, x):
        return self(x)

    def get_transform(self):
        return self.transform

    def get_linear_probing_embeddings(self, x):
        """Get embeddings for linear probing (CLS token)."""
        with torch.no_grad():
            return self.model(x)

    def get_segmentation_embeddings(self, x):
        """Get patch tokens for segmentation tasks."""
        with torch.no_grad():
            features = self.model.get_intermediate_layers(x, n=1)[0]
            return features


def run_thunder_benchmark(
    model_name: str = "openmidnight",
    checkpoint_path: str = None,
    datasets: list = None,
    tasks: list = None,
    loading_mode: str = "embedding_pre_loading",
    output_dir: str = None,
):
    """Run THUNDER benchmark evaluation."""

    try:
        from thunder import benchmark
        from thunder.models import PretrainedModel
    except ImportError:
        logger.error("THUNDER not installed. Install with: pip install thunder-bench")
        sys.exit(1)

    datasets = datasets or DATASETS_QUICK
    tasks = tasks or TASKS_CLASSIFICATION

    # Set up output directory
    if output_dir is None:
        output_dir = os.environ.get(
            "THUNDER_BASE_DATA_FOLDER",
            os.path.expanduser("~/thunder_results")
        )
    output_dir = Path(output_dir) / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {output_dir}")

    # Prepare model
    model_cls = None
    if checkpoint_path:
        # Use custom checkpoint
        logger.info(f"Using custom checkpoint: {checkpoint_path}")

        class CustomOpenMidnight(PretrainedModel):
            def __init__(self):
                super().__init__()
                self._model = OpenMidnightCustom(checkpoint_path)
                self.name = "openmidnight_custom"
                self.emb_dim = 1536
                self.vlm = False
                self.t = self._model.transform

            def forward(self, x):
                return self._model(x)

            def get_transform(self):
                return self.t

            def get_linear_probing_embeddings(self, x):
                return self._model.get_linear_probing_embeddings(x)

            def get_segmentation_embeddings(self, x):
                return self._model.get_segmentation_embeddings(x)

        model_cls = CustomOpenMidnight
        effective_model_name = "custom"
    else:
        effective_model_name = model_name

    # Run evaluations
    results = {}
    total = len(datasets) * len(tasks)
    completed = 0

    for dataset in datasets:
        results[dataset] = {}

        for task in tasks:
            completed += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"[{completed}/{total}] Evaluating: {effective_model_name} on {dataset} - {task}")
            logger.info(f"{'='*60}")

            try:
                if model_cls:
                    benchmark(
                        model_cls,
                        dataset=dataset,
                        task=task,
                        loading_mode=loading_mode,
                    )
                else:
                    benchmark(
                        model_name,
                        dataset=dataset,
                        task=task,
                        loading_mode=loading_mode,
                    )

                results[dataset][task] = {"status": "success"}
                logger.info(f"✓ SUCCESS: {dataset} - {task}")

            except Exception as e:
                results[dataset][task] = {"status": "failed", "error": str(e)}
                logger.error(f"✗ FAILED: {dataset} - {task}: {e}")

    # Save results summary
    results_file = output_dir / "results_summary.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*60}")

    passed = sum(1 for d in results.values() for t in d.values() if t["status"] == "success")
    failed = total - passed

    logger.info(f"Total: {total}, Passed: {passed}, Failed: {failed}")

    if failed > 0:
        logger.info("\nFailed evaluations:")
        for dataset, tasks_dict in results.items():
            for task, result in tasks_dict.items():
                if result["status"] == "failed":
                    logger.info(f"  - {dataset}/{task}: {result.get('error', 'Unknown error')}")

    logger.info(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="THUNDER Benchmark Evaluation for OpenMidnight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test on MHIST
    python thunder_evaluation.py --model openmidnight --dataset mhist --task simple_shot

    # Run all classification tasks on standard datasets
    python thunder_evaluation.py --model openmidnight --preset standard

    # Evaluate your own checkpoint
    python thunder_evaluation.py --checkpoint ./checkpoints/teacher_epoch250000.pth --dataset mhist

    # Compare few-shot vs linear probe performance
    python thunder_evaluation.py --model openmidnight --dataset mhist --task linear_probing simple_shot knn
        """
    )

    # Model options
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", "-m",
        type=str,
        default="openmidnight",
        help="Model name from THUNDER (default: openmidnight)"
    )
    model_group.add_argument(
        "--checkpoint", "-c",
        type=str,
        help="Path to custom checkpoint file"
    )

    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        nargs="+",
        help="Dataset(s) to evaluate on"
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=["quick", "standard", "full", "spider"],
        default="quick",
        help="Preset dataset configuration"
    )

    # Task options
    parser.add_argument(
        "--task", "-t",
        type=str,
        nargs="+",
        help="Task(s) to run"
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all classification tasks"
    )

    # Other options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--loading-mode",
        type=str,
        default="embedding_pre_loading",
        choices=["embedding_pre_loading", "image_pre_loading", "no_pre_loading"],
        help="Data loading mode"
    )

    args = parser.parse_args()

    # Determine datasets
    if args.dataset:
        datasets = args.dataset
    else:
        preset_map = {
            "quick": DATASETS_QUICK,
            "standard": DATASETS_STANDARD,
            "full": DATASETS_FULL,
            "spider": DATASETS_SPIDER,
        }
        datasets = preset_map[args.preset]

    # Determine tasks
    if args.all_tasks:
        tasks = TASKS_ALL
    elif args.task:
        tasks = args.task
    else:
        tasks = TASKS_CLASSIFICATION

    # Run evaluation
    run_thunder_benchmark(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        datasets=datasets,
        tasks=tasks,
        loading_mode=args.loading_mode,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
