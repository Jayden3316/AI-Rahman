import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from typing import Optional, Dict, Any, List
import os
import numpy as np
import json
from sklearn.decomposition import PCA
import torchaudio
import librosa


class MusicgenWithResiduals:
    """
    MusicGen model wrapper with hooks to capture residual streams from all layers.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize MusicGen model with hooks.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on
        """
        print(f"Loading model {model_name} to {device}...")
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_hidden_states=True
        ).to(device)

        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Freeze encoders
        self.model.freeze_text_encoder()
        self.model.freeze_audio_encoder()

        self.device = device
        self.hidden_states = {}

        def hook_fn(module, input, output):
            """Hook function to capture hidden states from decoder layers."""
            if hasattr(output, "hidden_states"):
                layer_names = []
                if hasattr(self.model, 'decoder') and hasattr(self.model.decoder.model.decoder, 'layers'):
                    layer_names += [f"decoder.layer.{i}" for i in range(len(self.model.decoder.model.decoder.layers))]
                
                self.hidden_states = {
                    layer_names[i]: output.hidden_states[i+1]
                    for i in range(len(layer_names))
                }
            else:
                print(f"Output structure: {type(output)} - {output}")

        # Register hook on decoder
        self.model.decoder.model.decoder.register_forward_hook(hook_fn)
        print("Model ready!")

    def generate_with_residuals(
        self,
        text: str = None,
        audio: Optional[torch.Tensor] = None,
        sampling_rate: int = None,
        max_new_tokens: int = 10,
        temperature: float = 1e-3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate audio with residual streams from all layers.
        
        Args:
            text: Text prompt
            audio: Input audio tensor
            sampling_rate: Audio sampling rate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing audio values and residual streams
        """
        self.model.decoder.config.output_hidden_states = True
        inputs = {}

        if text is None and audio is None:
            inputs = self.model.get_unconditional_inputs(num_samples=1)
        else:
            inputs = self.processor(
                text=text,
                audio=audio,
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )

        return {
            "audio_values": outputs.sequences,
            "residual_streams": self.hidden_states,
            "sampling_rate": self.model.config.audio_encoder.sampling_rate
        }


class VectorGuidedMusicgen(MusicgenWithResiduals):
    def __init__(self, model_name="facebook/musicgen-small", device="cuda"):
        super().__init__(model_name=model_name, device=device)
        self.steering_vectors = {}
        self.source_steering_vectors = {}
        self.target_steering_vectors = {}
        self.count = 0
        self.freq = 1
    def load_steering_vectors(self, steering_vectors, target_class, target_layers):
        """
        Load steering vectors from trained linear probes
        """
        for layer_idx in target_layers:
            steer = steering_vectors[layer_idx-1]
            steering_vector = steer[target_class]
            steering_vector = steering_vector / np.linalg.norm(steering_vector)
            self.steering_vectors[layer_idx] = torch.Tensor(steering_vector)

        print(f"Loaded steering vectors for layers: {target_layers}")
        print(f"Steering vector shape: {steering_vector.shape}")

    def load_multi_genre_steering(self, steering_vectors, source_class, target_class, target_layers):
        """
        Load steering vectors from trained linear probes
        Subtract the steering vector from the source class and add it to the target class
        adjust the ratio to control relative steering
        """
        for layer_idx in target_layers:
            steer = steering_vectors[layer_idx-1]
            source_vector = steer[source_class]
            target_vector = steer[target_class]
            self.source_steering_vectors[layer_idx] = source_vector / np.linalg.norm(source_vector)
            self.target_steering_vectors[layer_idx] = target_vector / np.linalg.norm(target_vector)

        print(f"Loaded steering vectors for layers: {target_layers}")

    def generate_with_multilayer_guidance(
        self,
        text: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        sampling_rate: int = None,
        max_new_tokens: int = 256,
        guidance_scale: float = 3.0,
        target_layers: List[int] = None,
        layer_strengths: Dict[int, float] = None,
        **kwargs
    ):
        if not target_layers:
            target_layers = list(self.steering_vectors.keys())
        if layer_strengths is None:
            print("Layer strengths not set, no steering is done")
            layer_strengths = {layer: 0.0 for layer in target_layers}

        hooks = []
        def create_layer_hook(layer_idx: int, strength: float):
            def residual_hook(module, input, output):
                self.count+=1 #delays the steering between layers if multiple are being steered
                original = output[0]
                #print(len(output))
                device = original.device
                steering_vector = self.steering_vectors[layer_idx].to(device)
                steering_vector = steering_vector.unsqueeze(0)  # [1, hidden_dim]

                cond = original[0]
                uncond = original[1]
                #print(cond.shape, uncond.shape)
                norm = torch.norm(original[0], dim=-1, keepdim=True)
                if self.count % self.freq == 0:
                    #print('steered successfully')
                    hidden_state = torch.stack([
                        (cond + norm * steering_vector * strength) / (1 + strength),
                        uncond
                    ])
                else:
                    hidden_state = torch.stack([
                        cond,
                        uncond
                    ])
                return (hidden_state).unsqueeze(0)

            return residual_hook

        for layer_idx in target_layers:
            strength = layer_strengths.get(layer_idx, 0.0)
            hook_fn = create_layer_hook(layer_idx, strength)
            layer_module = self.model.decoder.model.decoder.layers[layer_idx]
            hook = layer_module.register_forward_hook(hook_fn)
            hooks.append(hook)

        try:
            outputs = super().generate_with_residuals(
                text=text,
                audio=audio,
                sampling_rate=sampling_rate,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                **kwargs
            )
        finally:
            for hook in hooks:
                hook.remove()

        return outputs
    """
    ## TODO: Update this function to allow steering using multiple probes
    
    def generate_with_multilayer_multiprobe_guidance(
        self,
        text: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        sampling_rate: int = None,
        max_new_tokens: int = 256,
        guidance_scale: float = 3.0,
        target_layers: List[int] = None,
        layer_strengths: Dict[int, float] = None,
        **kwargs
    ):
        if not target_layers:
            target_layers = list(self.steering_vectors.keys())
        if layer_strengths is None:
            print("Layer strengths not set, no steering is done")
            layer_strengths = {layer: 0.0 for layer in target_layers}

        hooks = []
        def create_layer_hook(layer_idx: int, strength: float):
            def residual_hook(module, input, output):
                self.count+=1 #delays the steering between layers if multiple are being steered
                original = output[0]
                #print(len(output))
                device = original.device
                source_steering_vector = self.steering_vectors[layer_idx].to(device)

                steering_vector = steering_vector.unsqueeze(0)  # [1, hidden_dim]

                cond = original[0]
                uncond = original[1]
                #print(cond.shape, uncond.shape)
                norm = torch.norm(original[0], dim=-1, keepdim=True)
                if self.count%self.freq==0:
                    #print('steered successfully')
                    print('starting steering')
                    print(cond.shape, steering_vector.shape)
                    cos_sim = (torch.dot(cond[-1], steering_vector[-1]))/(torch.norm(cond)*torch.norm(steering_vector))
                    print(f'cos sim: {cos_sim}')
                    hidden_state = torch.stack([
                        (cond + norm * steering_vector *strength)/(1 + strength),
                        uncond
                    ])
                else:
                    hidden_state = torch.stack([
                        cond,
                        uncond
                    ])
                return (hidden_state).unsqueeze(0)

            return residual_hook

        for layer_idx in target_layers:
            strength = layer_strengths.get(layer_idx, 0.0)
            hook_fn = create_layer_hook(layer_idx, strength)
            layer_module = self.model.decoder.model.decoder.layers[layer_idx]
            hook = layer_module.register_forward_hook(hook_fn)
            hooks.append(hook)

        try:
            outputs = super().generate_with_residuals(
                text=text,
                audio=audio,
                sampling_rate=sampling_rate,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                **kwargs
            )
        finally:
            for hook in hooks:
                hook.remove()

        return outputs
    """
def load_weights_as_dict(weights_path):
    """
    Convert (24, 4, 1024) npy weights into dict {layer_idx: steering_vectors}.
    Each entry is shape (4, 1024) -> 4 genres per layer.
    """
    data = np.load(weights_path)   # shape (24, 4, 1024)

    weights_dict = {layer_idx+1: data[layer_idx] for layer_idx in range(data.shape[0])}
    return weights_dict


def steer_music(
    model,
    text: str = None,
    audio: torch.Tensor = None,
    sr: int = None,
    max_new_tokens: int = 256,
    guidance_scale: float = 3.0,
    target_layers: List[int] = None,
    layer_strengths: Dict[int, float] = None,
    steering_period: int = 25,
    offset: int = 0,
    weights_dict = None,
    source_class = None,
    target_class = None,
    target_sr = 32000,
    **kwargs
) -> Dict[str, Any]:
    """
    Steer music generation using trained steering vectors.
    """
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr

    # Re-initialize the model
    model = VectorGuidedMusicgen()

    model.count = offset
    model.freq = steering_period

    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0, keepdim=True).squeeze()
    print(f'audio shape: {audio.shape}')
    
    model.load_steering_vectors(weights_dict, target_class, target_layers)
    outputs = model.generate_with_multilayer_guidance(
        text=text,
        audio=[audio.squeeze(0).numpy()[sr*5:sr*10]],
        sampling_rate=target_sr,
        target_layers=target_layers,
        layer_strengths={i: 0.5 for i in target_layers} if layer_strengths is None else layer_strengths,
        guidance_scale=guidance_scale,
        max_new_tokens=max_new_tokens
    )
    return outputs

if __name__ == "__main__":
    # Example usage
    print("MusicGen hooks module loaded successfully!")
    print("Use MusicgenWithResiduals to capture activations from all layers.")
    print("Use VectorGuidedMusicgen for steering vector experiments.") 