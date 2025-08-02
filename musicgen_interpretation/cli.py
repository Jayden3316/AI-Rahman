"""
Command-line interface for MusicGen Interpretation Library.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from .musicgen_hooks import MusicgenWithResiduals, VectorGuidedMusicgen
from .linear_probes import train_probes_all_layers, evaluate_probe_performance, plot_results
from .data_processing import process_gtzan_batch, process_fma_batch, load_gtzan_dataset, load_fma_dataset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MusicGen Interpretation - Fine-grained control over Music Generation with Activation Steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process GTZAN dataset
  musicgen-interpretation process-gtzan --save-dir gtzan_processed --genres classical rock

  # Train linear probes
  musicgen-interpretation train-probes --data-dir gtzan_processed --output-dir probe_weights

  # Generate music with steering
  musicgen-interpretation generate --text "Generate classical music" --steering-weights probe_weights/mse_weights.npy
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process GTZAN command
    gtzan_parser = subparsers.add_parser('process-gtzan', help='Process GTZAN dataset')
    gtzan_parser.add_argument('--save-dir', required=True, help='Directory to save processed data')
    gtzan_parser.add_argument('--genres', nargs='+', help='Genres to process (default: all)')
    gtzan_parser.add_argument('--start-idx', type=int, default=0, help='Starting index')
    gtzan_parser.add_argument('--end-idx', type=int, help='Ending index')
    
    # Process FMA command
    fma_parser = subparsers.add_parser('process-fma', help='Process FMA dataset')
    fma_parser.add_argument('--save-dir', required=True, help='Directory to save processed data')
    fma_parser.add_argument('--genres', nargs='+', help='Genres to process (default: all)')
    fma_parser.add_argument('--max-samples', type=int, default=1000, help='Maximum samples to process')
    
    # Train probes command
    probe_parser = subparsers.add_parser('train-probes', help='Train linear probes')
    probe_parser.add_argument('--data-dir', required=True, help='Directory with processed data')
    probe_parser.add_argument('--output-dir', required=True, help='Directory to save probe weights')
    probe_parser.add_argument('--residual-type', choices=['conditional', 'unconditional'], 
                             default='conditional', help='Type of residual to use')
    probe_parser.add_argument('--loss-type', choices=['mse', 'cross_entropy'], 
                             default='mse', help='Loss function type')
    probe_parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    
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
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate probe performance')
    eval_parser.add_argument('--data-dir', required=True, help='Directory with processed data')
    eval_parser.add_argument('--weights-file', required=True, help='Path to probe weights file')
    eval_parser.add_argument('--output-plot', help='Path to save performance plot')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'process-gtzan':
            process_gtzan_command(args)
        elif args.command == 'process-fma':
            process_fma_command(args)
        elif args.command == 'train-probes':
            train_probes_command(args)
        elif args.command == 'generate':
            generate_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def process_gtzan_command(args):
    """Handle GTZAN processing command."""
    print("Loading MusicGen model...")
    model = MusicgenWithResiduals()
    
    print("Loading GTZAN dataset...")
    dataset = load_gtzan_dataset()
    
    print(f"Processing GTZAN data to {args.save_dir}...")
    process_gtzan_batch(
        model=model,
        dataset=dataset,
        save_dir=args.save_dir,
        genres_to_process=args.genres,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    print("GTZAN processing completed!")


def process_fma_command(args):
    """Handle FMA processing command."""
    print("Loading MusicGen model...")
    model = MusicgenWithResiduals()
    
    print("Loading FMA dataset...")
    dataset = load_fma_dataset()
    
    print(f"Processing FMA data to {args.save_dir}...")
    process_fma_batch(
        model=model,
        dataset=dataset,
        save_dir=args.save_dir,
        genres_to_process=args.genres,
        max_samples=args.max_samples
    )
    print("FMA processing completed!")


def train_probes_command(args):
    """Handle probe training command."""
    from .linear_probes import load_processed_data, save_weights
    
    print(f"Loading processed data from {args.data_dir}...")
    df = load_processed_data(args.data_dir)
    
    # Add labels (assuming binary classification: classical=-1, rock=1)
    genre_map = {'classical': -1, 'rock': 1}
    df['label'] = df['genre'].apply(lambda x: genre_map.get(x.lower(), 0))
    
    print(f"Training {args.loss_type} probes for all layers...")
    weights_dict = train_probes_all_layers(
        df=df,
        residual_type=args.residual_type,
        loss_type=args.loss_type,
        num_epochs=args.epochs
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    weights_file = os.path.join(args.output_dir, f"{args.loss_type}_weights.npy")
    save_weights(weights_dict, weights_file)
    print(f"Probe weights saved to {weights_file}")


def generate_command(args):
    """Handle music generation command."""
    import torch
    import soundfile as sf
    
    if args.steering_weights:
        print("Loading steering weights...")
        from .linear_probes import load_weights
        steering_vectors = load_weights(args.steering_weights)
        
        print("Loading guided MusicGen model...")
        model = VectorGuidedMusicgen()
        
        if args.target_layers:
            model.load_steering_vectors(steering_vectors, args.target_layers)
            
            layer_strengths = {}
            if args.layer_strengths:
                for i, layer in enumerate(args.target_layers):
                    layer_strengths[layer] = args.layer_strengths[i] if i < len(args.layer_strengths) else 0.5
            else:
                layer_strengths = {layer: 0.5 for layer in args.target_layers}
        else:
            layer_strengths = None
    else:
        print("Loading standard MusicGen model...")
        model = MusicgenWithResiduals()
        layer_strengths = None
    
    # Prepare inputs
    inputs = {}
    if args.text:
        inputs['text'] = args.text
    if args.audio:
        import librosa
        audio, sr = librosa.load(args.audio, sr=32000)
        inputs['audio'] = torch.tensor(audio)
        inputs['sampling_rate'] = sr
    
    print("Generating music...")
    if args.steering_weights and args.target_layers:
        outputs = model.generate_with_multilayer_guidance(
            **inputs,
            max_new_tokens=args.max_tokens,
            guidance_scale=args.guidance_scale,
            target_layers=args.target_layers,
            layer_strengths=layer_strengths
        )
    else:
        outputs = model.generate_with_residuals(
            **inputs,
            max_new_tokens=args.max_tokens,
            guidance_scale=args.guidance_scale
        )
    
    # Save audio
    audio_values = outputs['audio_values'].cpu().numpy()
    sf.write(args.output, audio_values, outputs['sampling_rate'])
    print(f"Generated audio saved to {args.output}")


def evaluate_command(args):
    """Handle probe evaluation command."""
    from .linear_probes import load_processed_data, load_weights, evaluate_probe_performance
    
    print(f"Loading processed data from {args.data_dir}...")
    df = load_processed_data(args.data_dir)
    
    # Add labels
    genre_map = {'classical': -1, 'rock': 1}
    df['label'] = df['genre'].apply(lambda x: genre_map.get(x.lower(), 0))
    
    print(f"Loading probe weights from {args.weights_file}...")
    weights_dict = load_weights(args.weights_file)
    
    print("Evaluating probe performance...")
    results = evaluate_probe_performance(df, weights_dict, 'conditional')
    
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