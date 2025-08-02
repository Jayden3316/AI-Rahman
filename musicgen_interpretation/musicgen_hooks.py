import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from typing import Optional, Dict, Any
import os
import numpy as np


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
    """
    Extended MusicGen model with steering vector capabilities.
    """
    
    def __init__(self, model_name="facebook/musicgen-small", device="cuda"):
        super().__init__(model_name=model_name, device=device)
        self.steering_vectors = {}
        self.count = 0
        self.freq = 18

    def load_steering_vectors(self, steering_vectors: Dict[int, np.ndarray], target_layers: list):
        """
        Load steering vectors from trained linear probes.
        
        Args:
            steering_vectors: Dictionary mapping layer indices to steering vectors
            target_layers: List of layer indices to apply steering to
        """
        for layer_idx in target_layers:
            steering_vector = steering_vectors[layer_idx-1]
            steering_vector = steering_vector / np.linalg.norm(steering_vector)
            self.steering_vectors[layer_idx] = torch.Tensor(steering_vector)

        print(f"Loaded steering vectors for layers: {target_layers}")
        print(f"Steering vector shape: {steering_vector.shape}")

    def generate_with_multilayer_guidance(
        self,
        text: Optional[str] = None,
        audio: Optional[torch.Tensor] = None,
        sampling_rate: int = None,
        max_new_tokens: int = 256,
        guidance_scale: float = 3.0,
        target_layers: list = None,
        layer_strengths: Dict[int, float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate audio with multi-layer steering guidance.
        
        Args:
            text: Text prompt
            audio: Input audio tensor
            sampling_rate: Audio sampling rate
            max_new_tokens: Maximum tokens to generate
            guidance_scale: Guidance scale for generation
            target_layers: List of layers to apply steering to
            layer_strengths: Dictionary mapping layer indices to steering strengths
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated audio and residual streams
        """
        if not target_layers:
            target_layers = list(self.steering_vectors.keys())
        if layer_strengths is None:
            print("Layer strengths not set, no steering is done")
            layer_strengths = {layer: 0.0 for layer in target_layers}

        hooks = []
        
        def create_layer_hook(layer_idx: int, strength: float):
            def residual_hook(module, input, output):
                self.count += 1
                original = output[0]
                device = original.device
                steering_vector = self.steering_vectors[layer_idx].to(device)
                steering_vector = steering_vector.unsqueeze(0)

                cond = original[0]
                uncond = original[1]
                norm = torch.norm(original[0], dim=-1, keepdim=True)
                
                if self.count % self.freq == 0:
                    hidden_state = torch.stack([
                        (cond + norm * steering_vector + strength) / (1 + strength),
                        uncond
                    ])
                else:
                    hidden_state = torch.stack([cond, uncond])

                return (hidden_state, output[1])

            return residual_hook

        # Register hooks for target layers
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
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return outputs


def save_activations(activations: Dict[str, torch.Tensor], save_path: str):
    """
    Save activations to .pth file.
    
    Args:
        activations: Dictionary of activations to save
        save_path: Path to save the file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activations, save_path)
    print(f"Saved activations to: {save_path}")


def load_activations(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load activations from .pth file.
    
    Args:
        load_path: Path to load the file from
        
    Returns:
        Dictionary of loaded activations
    """
    activations = torch.load(load_path)
    print(f"Loaded activations from: {load_path}")
    return activations


def extract_and_save_residuals(model: MusicgenWithResiduals, 
                              text: str = None,
                              audio: torch.Tensor = None,
                              sampling_rate: int = None,
                              save_path: str = "residuals.pth",
                              **kwargs) -> Dict[str, Any]:
    """
    Extract residual streams and save them to file.
    
    Args:
        model: MusicGen model with hooks
        text: Text prompt
        audio: Input audio tensor
        sampling_rate: Audio sampling rate
        save_path: Path to save residuals
        **kwargs: Additional generation parameters
        
    Returns:
        Dictionary containing generation outputs
    """
    outputs = model.generate_with_residuals(
        text=text,
        audio=audio,
        sampling_rate=sampling_rate,
        **kwargs
    )
    
    # Save residual streams
    save_activations(outputs['residual_streams'], save_path)
    
    return outputs


if __name__ == "__main__":
    # Example usage
    print("MusicGen hooks module loaded successfully!")
    print("Use MusicgenWithResiduals to capture activations from all layers.")
    print("Use VectorGuidedMusicgen for steering vector experiments.") 