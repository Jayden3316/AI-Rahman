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
    save_activations,
    load_activations,
    extract_and_save_residuals,
)

from .linear_probes import (
    SimpleNN,
    train_probe_mse,
    train_probe_cross_entropy,
    train_probes_all_layers,
    evaluate_probe_performance,
    plot_results,
    save_weights,
    load_weights,
    load_processed_data,
    prepare_data_for_training,
)

from .data_processing import (
    process_gtzan_data,
    process_fma_data,
    process_gtzan_batch,
    process_fma_batch,
    load_gtzan_dataset,
    load_fma_dataset,
    save_processed_data,
)

# Main exports
__all__ = [
    # MusicGen Hooks
    "MusicgenWithResiduals",
    "VectorGuidedMusicgen",
    "save_activations",
    "load_activations",
    "extract_and_save_residuals",
    
    # Linear Probes
    "SimpleNN",
    "train_probe_mse",
    "train_probe_cross_entropy",
    "train_probes_all_layers",
    "evaluate_probe_performance",
    "plot_results",
    "save_weights",
    "load_weights",
    "load_processed_data",
    "prepare_data_for_training",
    
    # Data Processing
    "process_gtzan_data",
    "process_fma_data",
    "process_gtzan_batch",
    "process_fma_batch",
    "load_gtzan_dataset",
    "load_fma_dataset",
    "save_processed_data",
] 