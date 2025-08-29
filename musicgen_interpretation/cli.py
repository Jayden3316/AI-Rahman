"""
Command-line interface for MusicGen Interpretation Library.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from .musicgen_hooks import MusicgenWithResiduals, VectorGuidedMusicgen, load_weights_as_dict, steer_music
from .linear_probes import train_probes_all_layers, evaluate_probe_performance, plot_results
from .data_processing import load_lewtun_dataset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MusicGen Interpretation - Fine-grained control over Music Generation with Activation Steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Lewtun dataset (4 genres: Classical, Electronic, Rock, Jazz)
  musicgen-interpretation process-lewtun --save-dir lewtun_processed

  # Train linear probes for 4-class classification
  musicgen-interpretation train-probes --data-dir lewtun_processed --output-dir probe_weights --num-classes 4

  # Generate music with steering
  musicgen-interpretation generate --text "Generate classical music" --steering-weights probe_weights/mse_weights.npy --target-class 0 --target-layers 19
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Process Lewtun command
    lewtun_parser = subparsers.add_parser('process-lewtun', help='Process Lewtun dataset')
    lewtun_parser.add_argument('--save-dir', required=True, help='Directory to save processed data')
    lewtun_parser.add_argument('--genres', nargs='+', help='Genres to process (default: all)')
    lewtun_parser.add_argument('--max-samples', type=int, default=500, help='Maximum samples to process')
    
    # Train probes command
    probe_parser = subparsers.add_parser('train-probes', help='Train linear probes')
    probe_parser.add_argument('--data-dir', required=True, help='Directory with processed data')
    probe_parser.add_argument('--output-dir', required=True, help='Directory to save probe weights')
    probe_parser.add_argument('--residual-type', choices=['conditional', 'unconditional'], 
                             default='conditional', help='Type of residual to use')
    probe_parser.add_argument('--loss-type', choices=['mse', 'cross_entropy'], 
                             default='mse', help='Loss function type')
    probe_parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    probe_parser.add_argument('--num-classes', type=int, default=4, help='Number of classes for classification')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate music with steering')
    generate_parser.add_argument('--text', help='Text prompt for generation')
    generate_parser.add_argument('--audio', help='Path to input audio file')
    generate_parser.add_argument('--output', required=True, help='Output audio file path')
    generate_parser.add_argument('--steering-weights', help='Path to steering weights file')
    generate_parser.add_argument('--target-layers', nargs='+', type=int, help='Target layers for steering')
    generate_parser.add_argument('--layer-strengths', nargs='+', type=float, help='Steering strengths for each layer')
    generate_parser.add_argument('--max-tokens', type=int, default=512, help='Maximum tokens to generate')
    generate_parser.add_argument('--guidance-scale', type=float, default=3.0, help='Guidance scale')
    generate_parser.add_argument('--target-class', type=int, default=2, help='Target class for steering')
    generate_parser.add_argument('--steering-period', type=int, default=25, help='Steering period')
    generate_parser.add_argument('--offset', type=int, default=0, help='Steering offset')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate probe performance')
    eval_parser.add_argument('--data-dir', required=True, help='Directory with processed data')
    eval_parser.add_argument('--weights-file', required=True, help='Path to probe weights file')
    eval_parser.add_argument('--output-plot', help='Path to save performance plot')
    eval_parser.add_argument('--num-classes', type=int, default=4, help='Number of classes for evaluation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'process-lewtun':
            process_lewtun_command(args)
        elif args.command == 'train-probes':
            train_probes_command(args)
        elif args.command == 'generate':
            generate_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def process_lewtun_command(args):
    """Handle Lewtun processing command."""
    print("Loading MusicGen model...")
    model = MusicgenWithResiduals()
    
    print("Loading Lewtun dataset...")
    dataset = load_lewtun_dataset()
    
    print(f"Processing Lewtun data to {args.save_dir}...")
    from .data_processing import save_activations
    save_activations(model, args.save_dir, dataset['train'])
    print("Lewtun processing completed!")


def train_probes_command(args):
    """Handle probe training command."""
    from .linear_probes import load_processed_data, save_weights
    
    print(f"Loading processed data from {args.data_dir}...")
    df = load_processed_data(args.data_dir)
    
    # Add labels for multi-class classification
    if args.num_classes == 4:
        # 4-class classification: Classical=0, Electronic=1, Rock=2, Jazz=3
        genre_map = {'Classical': 0, 'Electronic': 1, 'Rock': 2, 'Jazz': 3}
    else:
        # Binary classification: classical=-1, rock=1
        genre_map = {'classical': -1, 'rock': 1}
    
    df['label'] = df['genre'].apply(lambda x: genre_map.get(x, 0))
    
    # Balance dataset for multi-class
    if args.num_classes > 2:
        min_count = df.groupby('label').size().min()
        df = df.groupby('label').sample(n=min_count, random_state=42)
        print(f"Balanced dataset to {min_count} samples per class")
    
    print(f"Training {args.loss_type} probes for all layers...")
    weights_dict = train_probes_all_layers(
        df=df,
        residual_type=args.residual_type,
        loss_type=args.loss_type,
        num_epochs=args.epochs,
        num_classes=args.num_classes
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    weights_file = os.path.join(args.output_dir, f"{args.loss_type}_weights.npy")
    save_weights(weights_dict, weights_file)
    print(f"Probe weights saved to {weights_file}")


def generate_command(args):
    """Handle music generation command."""
    import torch
    import soundfile as sf
    import torchaudio
    
    if args.steering_weights:
        print("Loading steering weights...")
        weights_dict = load_weights_as_dict(args.steering_weights)
        
        if args.audio:
            print("Using steer_music function for audio input...")
            import librosa
            audio, sr = librosa.load(args.audio, sr=None)
            audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dimension
            
            outputs = steer_music(
                model=None,  # Will be created inside steer_music
                text=args.text,
                audio=audio_tensor,
                sr=sr,
                max_new_tokens=args.max_tokens,
                guidance_scale=args.guidance_scale,
                target_layers=args.target_layers or [19],
                layer_strengths={layer: 0.5 for layer in (args.target_layers or [19])},
                steering_period=args.steering_period,
                offset=args.offset,
                weights_dict=weights_dict,
                target_class=args.target_class,
                target_sr=32000
            )
        else:
            print("Loading guided MusicGen model...")
            model = VectorGuidedMusicgen()
            
            if args.target_layers:
                model.load_steering_vectors(weights_dict, args.target_class, args.target_layers)
                
                layer_strengths = {}
                if args.layer_strengths:
                    for i, layer in enumerate(args.target_layers):
                        layer_strengths[layer] = args.layer_strengths[i] if i < len(args.layer_strengths) else 0.5
                else:
                    layer_strengths = {layer: 0.5 for layer in args.target_layers}
            else:
                layer_strengths = None
            
            # Prepare inputs
            inputs = {}
            if args.text:
                inputs['text'] = args.text
            
            outputs = model.generate_with_multilayer_guidance(
                **inputs,
                max_new_tokens=args.max_tokens,
                guidance_scale=args.guidance_scale,
                target_layers=args.target_layers,
                layer_strengths=layer_strengths
            )
    else:
        print("Loading standard MusicGen model...")
        model = MusicgenWithResiduals()
        
        # Prepare inputs
        inputs = {}
        if args.text:
            inputs['text'] = args.text
        if args.audio:
            import librosa
            audio, sr = librosa.load(args.audio, sr=32000)
            inputs['audio'] = torch.tensor(audio)
            inputs['sampling_rate'] = sr
        
        outputs = model.generate_with_residuals(
            **inputs,
            max_new_tokens=args.max_tokens,
            guidance_scale=args.guidance_scale
        )
    
    # Save audio
    audio_values = outputs['audio_values'].cpu().numpy().reshape(-1)
    sf.write(args.output, audio_values, outputs['sampling_rate'])
    print(f"Generated audio saved to {args.output}")


def evaluate_command(args):
    """Handle probe evaluation command."""
    from .linear_probes import load_processed_data, load_weights, evaluate_probe_performance
    
    print(f"Loading processed data from {args.data_dir}...")
    df = load_processed_data(args.data_dir)
    
    # Add labels for multi-class classification
    if args.num_classes == 4:
        # 4-class classification: Classical=0, Electronic=1, Rock=2, Jazz=3
        genre_map = {'Classical': 0, 'Electronic': 1, 'Rock': 2, 'Jazz': 3}
    else:
        # Binary classification: classical=-1, rock=1
        genre_map = {'classical': -1, 'rock': 1}
    
    df['label'] = df['genre'].apply(lambda x: genre_map.get(x, 0))
    
    print(f"Loading probe weights from {args.weights_file}...")
    weights_dict = load_weights(args.weights_file)
    
    print("Evaluating probe performance...")
    results = evaluate_probe_performance(df, weights_dict, 'conditional', num_classes=args.num_classes)
    
    if args.output_plot:
        print(f"Saving performance plot to {args.output_plot}...")
        plot_results(
            list(range(1, 25)),
            results['accuracies'],
            results['losses'],
            save_path=args.output_plot
        )
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 