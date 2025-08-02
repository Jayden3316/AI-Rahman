# Fine-Grained control over Music Generation with Activation Steering

## Files Structure

1. **`data_processing.py`** - Handles data from different datasets (GTZAN and FMA)
2. **`musicgen_hooks.py`** - Loads MusicGen with hooks and saves activations
3. **`linear_probes.py`** - Trains linear probes and provides inference results
4. **`requirements.txt`** - Required dependencies
5. **`README.md`** - This file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Processing (`data_processing.py`)

This script handles data from GTZAN and FMA datasets with separate functions for each.

#### Key Functions:

- `process_gtzan_data(model, data, count)` - Process GTZAN dataset items
- `process_fma_data(model, data, count)` - Process FMA dataset items
- `process_gtzan_batch(model, dataset, save_dir, ...)` - Process batch of GTZAN data
- `process_fma_batch(model, dataset, save_dir, ...)` - Process batch of FMA data
- `load_gtzan_dataset()` - Load GTZAN dataset
- `load_fma_dataset()` - Load FMA dataset

#### Example Usage:
```python
from data_processing import *
from musicgen_hooks import MusicgenWithResiduals

# Load model
model = MusicgenWithResiduals()

# Load datasets
gtzan = load_gtzan_dataset()
fma = load_fma_dataset()

# Process GTZAN data
process_gtzan_batch(model, gtzan, "gtzan_processed", 
                   genres_to_process=['classical', 'rock'])

# Process FMA data
process_fma_batch(model, fma, "fma_processed", 
                 genres_to_process=['Classical', 'Rock'])
```

### 2. MusicGen Hooks (`musicgen_hooks.py`)

This script loads MusicGen with hooks and saves activations in .pth files.

#### Key Classes:

- `MusicgenWithResiduals` - Base class with hooks to capture residual streams
- `VectorGuidedMusicgen` - Extended class with steering vector capabilities

#### Key Functions:

- `save_activations(activations, save_path)` - Save activations to .pth file
- `load_activations(load_path)` - Load activations from .pth file
- `extract_and_save_residuals(model, ...)` - Extract and save residual streams

#### Example Usage:
```python
from musicgen_hooks import MusicgenWithResiduals, VectorGuidedMusicgen

# Basic model with hooks
model = MusicgenWithResiduals()

# Generate with residual capture
outputs = model.generate_with_residuals(
    text="Generate classical music",
    max_new_tokens=512
)

# Save activations
save_activations(outputs['residual_streams'], "residuals.pth")

# Vector guided model
guided_model = VectorGuidedMusicgen()
guided_model.load_steering_vectors(steering_vectors, [12])
outputs = guided_model.generate_with_multilayer_guidance(
    text="Generate rock music",
    target_layers=[12],
    layer_strengths={12: 0.6}
)
```

### 3. Linear Probes (`linear_probes.py`)

This script trains linear probes and provides inference results with both MSE and Cross Entropy loss.

#### Key Functions:

- `train_probe_mse(X, y, ...)` - Train probe using MSE loss (without sigmoid/softmax)
- `train_probe_cross_entropy(X, y, ...)` - Train probe using Cross Entropy loss
- `train_probes_all_layers(df, residual_type, loss_type, ...)` - Train probes for all layers
- `evaluate_probe_performance(df, weights_dict, ...)` - Evaluate probe performance
- `plot_results(layers, accuracies, losses, ...)` - Plot training results

#### Example Usage:
```python
from linear_probes import *
import pandas as pd

# Load processed data
df = load_processed_data("gtzan_processed")

# Add labels (assuming binary classification: classical=-1, rock=1)
genre_map = {'classical': -1, 'rock': 1}
df['label'] = df['genre'].apply(lambda x: genre_map[x])

# Train MSE probes for all layers
weights_dict_mse = train_probes_all_layers(
    df, 
    residual_type='conditional',
    loss_type='mse',
    num_epochs=250
)

# Train Cross Entropy probes for all layers
weights_dict_ce = train_probes_all_layers(
    df, 
    residual_type='conditional',
    loss_type='cross_entropy',
    num_epochs=250
)

# Evaluate performance
results = evaluate_probe_performance(df, weights_dict_mse, 'conditional')

# Plot results
plot_results(
    list(range(1, 25)), 
    results['accuracies'], 
    results['losses'],
    save_path="probe_results.png"
)

# Save weights
save_weights(weights_dict_mse, "mse_weights.npy")
save_weights(weights_dict_ce, "ce_weights.npy")
```

## Key Features

### Data Processing
- Separate functions for GTZAN and FMA datasets
- Audio resampling to 32kHz
- Residual stream extraction from all 24 layers
- Batch processing capabilities
- Error handling and progress tracking

### MusicGen Hooks
- Captures residual streams from all decoder layers
- Supports both conditional and unconditional generation
- Vector steering capabilities for controlled generation
- Activation saving/loading functionality

### Linear Probes
- **MSE Loss Training**: Trains without sigmoid/softmax as requested
- **Cross Entropy Loss Training**: Standard classification training
- Layer-wise training for all 24 layers
- Performance evaluation and visualization
- Weight saving/loading functionality

## Notes

- The MSE loss training is done without sigmoid/softmax as specified
- Both loss functions provide accuracy and loss metrics
- The code supports both GTZAN and FMA datasets with different processing functions
- All activations are saved in .pth format for easy loading
- The vector steering functionality allows for controlled music generation

## Dependencies

See `requirements.txt` for the complete list of required packages. The main dependencies are:
- PyTorch for deep learning
- Transformers for MusicGen model
- Librosa for audio processing
- Scikit-learn for data preprocessing
- Datasets for loading GTZAN and FMA
- Matplotlib for visualization 