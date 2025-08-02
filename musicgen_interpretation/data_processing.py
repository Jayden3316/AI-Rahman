import numpy as np
import librosa
import os
import re
from datasets import load_dataset
from typing import Dict, List, Any
import torch


def sanitize_string(s: str) -> str:
    """Sanitize string by removing non-alphanumeric characters."""
    return re.sub(r'[^a-zA-Z0-9]', '', s)


def process_gtzan_data(model, data: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """
    Process GTZAN dataset with MusicGen model to extract residual streams.
    
    Args:
        model: MusicGen model with hooks
        data: GTZAN dataset item
        count: Index counter
    
    Returns:
        List of processed results with residual streams
    """
    target_sr = 32000
    audio = data['audio']
    sr = audio['sampling_rate']
    
    # Get genre label
    id2label_fn = data['genre'].__class__.int2str
    genre = id2label_fn(data['genre'])
    prompt = f"Generate {genre} music continuing the given audio"
    
    # Resample audio
    audio['array'] = librosa.resample(y=audio['array'], orig_sr=sr, target_sr=target_sr)
    
    # Create audio segments
    audio_segments = [
        audio['array'][i*target_sr:(i+10)*target_sr] 
        for i in range(5, int(audio['array'].shape[0]//target_sr), 5)
    ]
    
    result = []
    for segment in audio_segments:
        outputs = model.generate_with_residuals(
            text=prompt,
            audio=segment,
            sampling_rate=target_sr,
            max_new_tokens=512,
            guidance_scale=3.0,
            do_sample=True
        )
        
        # Get residual stream from all layers
        residual = np.array([
            outputs['residual_streams'][i].detach().cpu().numpy() 
            for i in outputs['residual_streams']
        ])
        
        # Create result dictionary
        result.append({
            'genre': genre,
            'generated_audio': outputs['audio_values'].detach().cpu().numpy(),
            'residual_stream': residual,
            'sampling_rate': outputs['sampling_rate'],
            'prompt_used': prompt
        })
    
    return result


def process_fma_data(model, data: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """
    Process FMA Medium dataset with MusicGen model to extract residual streams.
    
    Args:
        model: MusicGen model with hooks
        data: FMA dataset item
        count: Index counter
    
    Returns:
        List of processed results with residual streams
    """
    target_sr = 32000
    audio = data['audio']
    sr = audio['sampling_rate']
    
    # Get genre label from FMA
    id2label_fma = data['genres'].__class__.feature.int2str
    genre = id2label_fma(data['genres'][0])
    prompt = f"Generate {genre} music continuing the given audio"
    
    # Resample audio
    audio['array'] = librosa.resample(y=audio['array'], orig_sr=sr, target_sr=target_sr)
    
    # Create audio segments (different segmentation for FMA)
    audio_segments = [
        audio['array'][i*target_sr:(i+10)*target_sr] 
        for i in range(0, int(audio['array'].shape[0]//target_sr), 10)
    ]
    
    result = []
    for segment in audio_segments:
        outputs = model.generate_with_residuals(
            text=prompt,
            audio=segment,
            sampling_rate=target_sr,
            max_new_tokens=512,
            guidance_scale=3.0,
            do_sample=True
        )
        
        # Get residual stream from all layers
        residual = np.array([
            outputs['residual_streams'][i].detach().cpu().numpy() 
            for i in outputs['residual_streams']
        ])
        
        # Create result dictionary
        result.append({
            'genre': genre,
            'generated_audio': outputs['audio_values'].detach().cpu().numpy(),
            'residual_stream': residual,
            'sampling_rate': outputs['sampling_rate'],
            'prompt_used': prompt
        })
    
    return result


def save_processed_data(processed_data: List[Dict[str, Any]], save_dir: str, 
                       filename: str) -> None:
    """
    Save processed data to .npz files.
    
    Args:
        processed_data: List of processed data dictionaries
        save_dir: Directory to save files
        filename: Base filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, result in enumerate(processed_data):
        save_path = os.path.join(save_dir, f"{filename}_{i}.npz")
        np.savez(save_path, **result)
        print(f"Saved: {save_path}")


def load_gtzan_dataset():
    """Load GTZAN dataset."""
    return load_dataset("marsyas/gtzan", trust_remote_code=True)


def load_fma_dataset():
    """Load FMA Medium dataset."""
    return load_dataset("benjamin-paine/free-music-archive-medium", 
                       trust_remote_code=True, streaming=True)


def process_gtzan_batch(model, dataset, save_dir: str, genres_to_process: List[str] = None,
                       start_idx: int = 0, end_idx: int = None):
    """
    Process a batch of GTZAN data and save results.
    
    Args:
        model: MusicGen model with hooks
        dataset: GTZAN dataset
        save_dir: Directory to save processed data
        genres_to_process: List of genres to process (if None, process all)
        start_idx: Starting index
        end_idx: Ending index (if None, process to end)
    """
    if end_idx is None:
        end_idx = len(dataset['train'])
    
    for idx in range(start_idx, end_idx):
        data = dataset['train'][idx]
        genre = sanitize_string(data['genre'].__class__.int2str(data['genre']))
        
        # Filter by genre if specified
        if genres_to_process and genre not in genres_to_process:
            continue
            
        try:
            processed_data = process_gtzan_data(model, data, idx)
            filename = f"{genre}_{idx}"
            save_processed_data(processed_data, save_dir, filename)
            
            if idx % 10 == 0:
                print(f"Processed {idx} GTZAN samples")
                
        except Exception as e:
            print(f"Error processing GTZAN sample {idx}: {e}")
            continue


def process_fma_batch(model, dataset, save_dir: str, genres_to_process: List[str] = None,
                     max_samples: int = 1000):
    """
    Process a batch of FMA data and save results.
    
    Args:
        model: MusicGen model with hooks
        dataset: FMA dataset
        save_dir: Directory to save processed data
        genres_to_process: List of genres to process (if None, process all)
        max_samples: Maximum number of samples to process
    """
    count = 0
    for data in dataset['train']:
        if count >= max_samples:
            break
            
        genre = data['genres'].__class__.feature.int2str(data['genres'][0])
        
        # Filter by genre if specified
        if genres_to_process and genre not in genres_to_process:
            continue
            
        try:
            processed_data = process_fma_data(model, data, count)
            filename = f"{genre}_{count}"
            save_processed_data(processed_data, save_dir, filename)
            
            if count % 10 == 0:
                print(f"Processed {count} FMA samples")
                
            count += 1
            
        except Exception as e:
            print(f"Error processing FMA sample {count}: {e}")
            continue


if __name__ == "__main__":
    # Example usage
    print("Data processing module loaded successfully!")
    print("Use the functions to process GTZAN and FMA datasets with MusicGen model.") 