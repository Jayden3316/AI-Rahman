"""
MusicGen Interpretation Library

Fine-grained control over Music Generation with Activation Steering.
"""

__version__ = "0.1.0"
__author__ = "MusicGen Interpretation Team"
__email__ = "team@example.com"

# Import main classes and functions
from .musicgen_hooks import (
    MusicgenWithResiduals,
    VectorGuidedMusicgen,
    load_weights_as_dict,
    steer_music,
)

from .linear_probes import (
    SimpleNN,
    sign,
    load_processed_data,
    prepare_data_for_training,
    train_probe_mse,
    train_probe_cross_entropy,
    train_probes_all_layers,
    evaluate_probe_performance,
    plot_results,
    save_weights,
    load_weights,
)

from .data_processing import (
    sanitize_string,
    process_with_residuals,
    save_activations,
    load_lewtun_dataset,
    get_data,
)

# CLI entry point
from .cli import main as cli_main

# Main exports
__all__ = [
    # MusicGen Hooks
    "MusicgenWithResiduals",
    "VectorGuidedMusicgen",
    "load_weights_as_dict",
    "steer_music",
    
    # Linear Probes
    "SimpleNN",
    "sign",
    "load_processed_data",
    "prepare_data_for_training",
    "train_probe_mse",
    "train_probe_cross_entropy",
    "train_probes_all_layers",
    "evaluate_probe_performance",
    "plot_results",
    "save_weights",
    "load_weights",
    
    # Data Processing
    "sanitize_string",
    "process_with_residuals",
    "save_activations",
    "load_lewtun_dataset",
    "get_data",
    
    # CLI
    "cli_main",
]