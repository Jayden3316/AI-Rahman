# Installation Guide for MusicGen Interpretation Library

This guide explains how to install and use the MusicGen Interpretation Library for fine-grained control over music generation with activation steering.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- At least 8GB RAM (16GB recommended)

## Installation Options

### Option 1: Install from Source (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/musicgen-interpretation.git
   cd musicgen-interpretation
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   ```

   This installs the package in "editable" mode, allowing you to modify the code and see changes immediately.

### Option 2: Install with pip

```bash
pip install musicgen-interpretation
```

### Option 3: Install with conda

```bash
conda install -c conda-forge musicgen-interpretation
```

## Dependencies

The library automatically installs the following dependencies:

- **PyTorch** (>=2.0.0) - Deep learning framework
- **Transformers** (>=4.30.0) - HuggingFace transformers library
- **Librosa** (>=0.10.0) - Audio processing
- **Scikit-learn** (>=1.3.0) - Machine learning utilities
- **Soundfile** (>=0.12.0) - Audio file I/O
- **Datasets** (>=2.14.0) - Dataset loading
- **NumPy** (>=1.24.0) - Numerical computing
- **Matplotlib** (>=3.7.0) - Plotting
- **Pandas** (>=2.0.0) - Data manipulation

## Optional Dependencies

For development and testing:

```bash
pip install -e ".[dev]"
```

For documentation:

```bash
pip install -e ".[docs]"
```

## Quick Start

### 1. Basic Usage

```python
from musicgen_interpretation import MusicgenWithResiduals

# Initialize model
model = MusicgenWithResiduals()

# Generate music with residual capture
outputs = model.generate_with_residuals(
    text="Generate classical music with piano",
    max_new_tokens=256
)

# Access generated audio and residual streams
audio = outputs['audio_values']
residuals = outputs['residual_streams']
```

### 2. Command Line Interface

The library provides a comprehensive CLI for common tasks:

```bash
# Process GTZAN dataset
musicgen-interpretation process-gtzan --save-dir gtzan_processed --genres classical rock

# Train linear probes
musicgen-interpretation train-probes --data-dir gtzan_processed --output-dir probe_weights

# Generate music with steering
musicgen-interpretation generate --text "Generate rock music" --steering-weights probe_weights/mse_weights.npy --output rock_output.wav

# Evaluate probe performance
musicgen-interpretation evaluate --data-dir gtzan_processed --weights-file probe_weights/mse_weights.npy
```

### 3. Advanced Usage

```python
from musicgen_interpretation import VectorGuidedMusicgen, load_weights

# Load trained steering vectors
steering_vectors = load_weights("probe_weights/mse_weights.npy")

# Initialize guided model
model = VectorGuidedMusicgen()
model.load_steering_vectors(steering_vectors, [12, 18])

# Generate with steering
outputs = model.generate_with_multilayer_guidance(
    text="Generate jazz music",
    target_layers=[12, 18],
    layer_strengths={12: 0.5, 18: 0.3}
)
```

## Examples

Run the provided examples:

```bash
# Run basic usage examples
python examples/basic_usage.py
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce `max_new_tokens` or `batch_size`
   - Use CPU if GPU memory is insufficient

2. **Model loading issues:**
   - Ensure you have sufficient disk space for model downloads
   - Check internet connection for model downloads

3. **Audio processing errors:**
   - Ensure audio files are in supported formats (WAV, MP3, etc.)
   - Check audio file integrity

### Getting Help

- Check the [README.md](README.md) for detailed documentation
- Run examples to verify installation
- Check the [issues page](https://github.com/your-username/musicgen-interpretation/issues) for known problems

## Development Setup

For developers who want to contribute:

1. **Fork the repository**

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Run tests:**
   ```bash
   pytest
   ```

5. **Format code:**
   ```bash
   black musicgen_interpretation/
   isort musicgen_interpretation/
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{musicgen_interpretation,
  title={MusicGen Interpretation: Fine-grained control over Music Generation with Activation Steering},
  author={MusicGen Interpretation Team},
  year={2024},
  url={https://github.com/your-username/musicgen-interpretation}
}
``` 