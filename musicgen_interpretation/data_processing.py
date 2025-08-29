# get the residuals from given input samples
import numpy as np
import librosa
import re
import os
from datasets import load_dataset
from typing import Optional, List, Dict, Any


def sanitize_string(s):
    return re.sub(r'[^a-zA-Z0-9]', '', s)

def process_with_residuals(model, data, count):
    target_sr = 32000
    #print(data)
    audio = np.array(data['audio'][0]['array'])
    sr = data['audio'][0]['sampling_rate']
    genre = data['genre'][0]
    prompt = f"Generate {genre} music continuing the given audio"
    #print(audio['array'].shape)
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    audio_segments = [audio[i*target_sr:(i+10)*target_sr] for i in range(0, int(audio.shape[0]//target_sr), 10)]
    result = []
    for segment in audio_segments:
        #print(f"Segment shape: {segment.shape} segment type: {type(segment)}")
        outputs = model.generate_with_residuals(
            text=prompt,
            audio=segment,
            sampling_rate=target_sr,
            max_new_tokens=512,
            guidance_scale=3.0,
            do_sample=True
        )
        print('generated outputs')
        # Get the residual stream from the last layer
        residual = np.array([outputs['residual_streams'][i].detach().cpu().numpy() for i in outputs['residual_streams']])
        
        # Create result dictionary with all original features and new data
        result.append({
            'genre': genre,
            'generated_audio': outputs['audio_values'].detach().cpu().numpy(),
            'residual_stream': residual,
            'sampling_rate': outputs['sampling_rate'],
            'prompt_used': prompt
        })
    
    return result

def save_activations(
    model,
    save_dir,
    dataset,
):
    os.makedirs(save_dir, exist_ok=True)
    print('Made folder')
    idx = 0
    for data in dataset.iter(batch_size=1):
        genre = data['genre'][0]
        name = f"{genre}_{idx}"
        processed_data = process_with_residuals(model, data, idx)
        for result in range(len(processed_data)):
            print(f"processed {name}_{result}")
            # Save using numpy's save function
            save_path = os.path.join(save_dir, f"{name}_{result}.npz")
            np.savez(
                save_path,
                **processed_data[result]
            )
            
            if idx % 1 == 0:  # Print progress every 10 items
                print(f"Processed {idx} samples")
        idx += 1

def load_lewtun_dataset():
    """Load Lewtun modified dataset from HuggingFace."""
    try:
        dataset = load_dataset("roovy54/lewtun_music_genres_modified", streaming=True)
        return dataset
    except Exception as e:
        print(f"Error loading Lewtun dataset: {e}")
        return None

def get_data(save_dir: str) -> List[Dict[str, Any]]:
    """
    Load processed data from .npz files.
    
    Args:
        save_dir: Directory containing processed .npz files (activations from the residual stream)
        
    Returns:
        List of dictionaries with loaded data (which can be used as a dataset to train the probes)
    """
    data = []
    # Iterate through each file in the directory
    for filename in os.listdir(save_dir):
        if filename.endswith('.npz'):
            # Load the file
            loaded_file = dict(np.load(os.path.join(save_dir, filename), allow_pickle=True))

            # Reshape the residual streams
            if not data:
                print("Residual shape:", loaded_file['residual_stream'].shape)
            residual_unconditional = loaded_file['residual_stream'].reshape(24, 2, 1024)[:, 1, :]
            residual_conditional = loaded_file['residual_stream'].reshape(24, 2, 1024)[:, 0, :]

            # Create a dictionary for the current file's data
            file_data = {
                'filename': filename,
                'genre': loaded_file['genre'],  # Keeping original genre if needed
            }

            # Add residuals to the dictionary
            for layer in range(24):
                file_data[f'residual_conditional_{layer + 1}'] = residual_conditional[layer, :]
                file_data[f'residual_unconditional_{layer + 1}'] = residual_unconditional[layer, :]

            # Append the file data to the list
            data.append(file_data)

    return data