# MusicGen Interpretation

Fine-grained control over Music Generation with Activation Steering using Facebook's MusicGen model.

## Overview

- Extract and analyze hidden states from all decoder layers
- Train classifiers on internal representations for genre classification
- Guide music generation using trained steering vectors
- Support for 4-genre classification (Classical, Electronic, Rock, Jazz)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB of GPU memory for small models

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install the Library

```bash
pip install -e .
```

## Quick Start

### 1. Process Dataset

Process the Lewtun music genres dataset to extract residual streams:

```bash
python -m musicgen_interpretation.cli process-lewtun --save-dir data/processed
```

### 2. Train Linear Probes

Train linear probes for 4-class genre classification:

```bash
python -m musicgen_interpretation.cli train-probes \
    --data-dir data/processed \
    --output-dir models/probes \
    --num-classes 4 \
    --epochs 250
```

### 3. Generate Steered Music

Generate music with steering towards a specific genre:

```bash
python -m musicgen_interpretation.cli generate \
    --text "Generate energetic music" \
    --steering-weights models/probes/mse_weights.npy \
    --target-class 2 \
    --target-layers 19 \
    --output generated_rock.wav
```

## Usage Examples

### Python API

```python
import torch
from musicgen_interpretation import MusicgenWithResiduals, VectorGuidedMusicgen, load_weights_as_dict

# Load model with residual extraction
model = MusicgenWithResiduals()

# Generate music and extract residuals
outputs = model.generate_with_residuals(
    text="Generate classical music",
    max_new_tokens=512
)

# Access residual streams from all layers
residuals = outputs['residual_streams']
audio = outputs['audio_values']

# For steering, load trained weights
weights_dict = load_weights_as_dict("models/probes/mse_weights.npy")
steering_model = VectorGuidedMusicgen()

# Load steering vectors for specific genre (e.g., Rock = class 2)
steering_model.load_steering_vectors(weights_dict, target_class=2, target_layers=[19])

# Generate with steering
steered_output = steering_model.generate_with_multilayer_guidance(
    text="Generate music",
    target_layers=[19],
    layer_strengths={19: 0.5}
)
```

### Advanced Steering with Audio Input

```python
import torchaudio
from musicgen_interpretation import steer_music, load_weights_as_dict

# Load audio file
audio, sr = torchaudio.load("input.mp3")
weights_dict = load_weights_as_dict("models/probes/mse_weights.npy")

# Steer music generation based on input audio
outputs = steer_music(
    model=None,  # Will be created internally
    text="Continue this music in rock style",
    audio=audio,
    sr=sr,
    target_class=2,  # Rock
    target_layers=[19],
    steering_period=25,
    weights_dict=weights_dict
)
```

## Command Line Interface

### Available Commands

- `process-lewtun`: Process Lewtun dataset and extract residual streams
- `train-probes`: Train linear probes for genre classification
- `generate`: Generate music with optional steering
- `evaluate`: Evaluate probe performance across layers

### Detailed Command Usage

#### Process Dataset
```bash
python -m musicgen_interpretation.cli process-lewtun \
    --save-dir data/processed \
    --max-samples 500
```

#### Train Probes
```bash
python -m musicgen_interpretation.cli train-probes \
    --data-dir data/processed \
    --output-dir models/probes \
    --num-classes 4 \
    --loss-type mse \
    --epochs 250
```

#### Generate Music
```bash
python -m musicgen_interpretation.cli generate \
    --text "Generate energetic electronic music" \
    --steering-weights models/probes/mse_weights.npy \
    --target-class 1 \
    --target-layers 19 20 21 \
    --layer-strengths 0.5 0.3 0.2 \
    --steering-period 25 \
    --output electronic_music.wav
```

#### Evaluate Probes
```bash
python -m musicgen_interpretation.cli evaluate \
    --data-dir data/processed \
    --weights-file models/probes/mse_weights.npy \
    --num-classes 4 \
    --output-plot results/performance.png
```

## Model Architecture

### Supported Models
- `facebook/musicgen-small` (default)
- `facebook/musicgen-medium`
- `facebook/musicgen-large`

### Genre Classes
- **0**: Classical
- **1**: Electronic  
- **2**: Rock
- **3**: Jazz

### Layer Architecture
- **24 Decoder Layers**: Each layer's residual stream can be analyzed and steered
- **1024 Hidden Dimensions**: Feature vectors for classification and steering
- **Conditional/Unconditional Streams**: Separate analysis of guided vs unguided generation

## Notebooks

The `test/` directory contains Jupyter notebooks demonstrating key functionality:

- `steering_musicgen-update.ipynb`: Complete steering workflow
- `musicgen-interp-update.ipynb`: Basic interpretation examples  
- `probe-training-update.ipynb`: Linear probe training examples

## File Structure

```
musicgen_interpretation/
├── __init__.py              # Main library exports
├── cli.py                   # Command-line interface
├── musicgen_hooks.py        # MusicGen model wrappers with hooks
├── linear_probes.py         # Linear probe training and evaluation
├── data_processing.py       # Dataset processing utilities
├── main.py                  # Entry point
└── test/                    # Example notebooks
    ├── steering_musicgen-update.ipynb
    ├── musicgen-interp-update.ipynb
    └── probe-training-update.ipynb
```

## Performance Tips

### GPU Memory Management
- Use `torch.cuda.empty_cache()` between experiments
- Start with `musicgen-small` for development TODO: some parts have been hardcoded to include only 24 layers while `musicgen-medium` and `musicgen-large` have 48. A future version will make this to be model agnostic.
- Monitor GPU memory usage with large batch sizes

### Training Recommendations
- Use stratified sampling for balanced datasets
- Start with MSE loss for initial experiments
- Layer 17-21 typically show best steering performance (As mentioned in the main text, this is determined using the accuracy of the linear probes)
- Steering strength 0.3-0.7 works well for most cases

### Audio Quality
- Use 32kHz sampling rate for best results. `musicgen-small` was trained at this sampling rate. Other sampling rates are highly likely to throw errors.
- Generate 5-10 second clips for experimentation

## Acknowledgments

- Facebook AI Research for the MusicGen model
- lewtun https://huggingface.co/lewtun for the dataset that we used to collect activations from different genres.
