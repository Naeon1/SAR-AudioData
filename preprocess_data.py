"""
Data preprocessing script.
Converts audio data into Mel-spectrogram features and saves them as .npy files
to avoid recomputing features during training.
"""
import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from models.dataset import AudioEventDataset

from datetime import datetime
from collections import Counter
import torch

def preprocess_and_save(
    audio_dir: str,
    annotation_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    n_mels: int = 40,
    window_size: float = 3.0,
    hop_size: float = 1.0,
    event_type: str = None
):
    """
    Preprocess the data and save it.

    Args:
        audio_dir: Directory of audio files
        annotation_dir: Directory of annotation files
        output_dir: Output directory
        sample_rate: Sample rate
        n_mels: Number of Mel filter banks
        window_size: Audio window size (seconds)
        hop_size: Window hop size (seconds)
        event_type: Target event type
    """
    print("Starting data preprocessing...")
    print(f"Audio directory: {audio_dir}")
    print(f"Annotation directory: {annotation_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: {sample_rate}")
    print(f"Number of Mel bands: {n_mels}")
    print(f"Window size: {window_size}s")
    print(f"Hop size: {hop_size}s")
    print(f"Event type: {event_type}")

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create the dataset (this loads all audio and annotations and builds windows)
    print("\nLoading raw dataset...")
    dataset = AudioEventDataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        sample_rate=sample_rate,
        n_mels=n_mels,
        window_size=window_size,
        hop_size=hop_size,
        event_type=event_type
    )

    if len(dataset) == 0:
        print("Error: dataset is empty, please check the audio and annotation file paths")
        return

    print(f"Dataset size: {len(dataset)} samples")

    # Preprocess all data
    print("\nExtracting features and saving...")
    features_list = []
    labels_list = []
    label_stats = Counter()  # Used to collect label statistics

    for idx in tqdm(range(len(dataset)), desc="Processing"):
        try:
            # Get features and labels (Mel spectrogram is computed here)
            features, labels = dataset[idx]

            # Convert to numpy arrays and append to the lists
            features_list.append(features.numpy())
            labels_list.append(labels.numpy())

            # Collect label statistics
            if isinstance(labels, torch.Tensor):
                labels_np = labels.numpy()
            else:
                labels_np = labels

            # Count the occurrences of each label
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_stats[label] += count

        except Exception as e:
            print(f"\nWarning: error while processing sample {idx}: {e}")
            continue

    if len(features_list) == 0:
        print("Error: no samples were processed successfully")
        return

    # Convert to numpy arrays
    print("\nConverting to numpy arrays...")
    features_array = np.array(features_list, dtype=object)  # Use object dtype for variable-length sequences
    labels_array = np.array(labels_list, dtype=object)

    # Save as .npy files
    features_path = os.path.join(output_dir, 'features.npy')
    labels_path = os.path.join(output_dir, 'labels.npy')

    print(f"\nSaving features to: {features_path}")
    np.save(features_path, features_array, allow_pickle=True)

    print(f"Saving labels to: {labels_path}")
    np.save(labels_path, labels_array, allow_pickle=True)

    # Compute label statistics
    total_labels = int(sum(label_stats.values()))  # Convert to Python int
    label_distribution = {str(label): int(count) for label, count in label_stats.items()}  # Convert to Python int
    label_percentages = {str(label): f"{(int(count)/total_labels)*100:.2f}%" for label, count in label_stats.items()}

    # Save the configuration
    config = {
        'audio_dir': audio_dir,
        'annotation_dir': annotation_dir,
        'sample_rate': sample_rate,
        'n_mels': n_mels,
        'window_size': window_size,
        'hop_size': hop_size,
        'event_type': event_type,
        'num_samples': len(features_list),
        'feature_shape_info': f"variable-length sequence, channels=1, Mel bands={n_mels}",
        'label_statistics': {
            'total_labels': total_labels,
            'label_distribution': label_distribution,
            'label_percentages': label_percentages,
            'num_unique_labels': len(label_stats)
        }
    }

    config_path = os.path.join(output_dir, 'preprocess_config.json')
    print(f"Saving configuration to: {config_path}")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # Print statistics
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)
    print(f"Total samples: {len(features_list)}")
    print(f"Feature array shape: {features_array.shape}")
    print(f"Label array shape: {labels_array.shape}")

    # Print label statistics
    print(f"\nLabel statistics:")
    print(f"Total labels: {total_labels}")
    print(f"Unique labels: {len(label_stats)}")
    print("Label distribution:")
    for label, count in sorted(label_stats.items()):
        percentage = (count/total_labels)*100
        print(f"  Label {label}: {count} times ({percentage:.2f}%)")

    # Print the shape of a few sample entries
    if len(features_list) > 0:
        print(f"\nExample sample shapes:")
        for i in range(min(3, len(features_list))):
            print(f"  Sample {i}: features {features_list[i].shape}, labels {labels_list[i].shape}")

    print(f"\nPreprocessed data saved to: {output_dir}")
    print("During training, use the PreprocessedAudioDataset class to load this data")


def main():
    parser = argparse.ArgumentParser(description="Audio data preprocessing")
    parser.add_argument("--data_dir", type=str, default="data", help="Data root directory")

    parser.add_argument("--audio_subdir", type=str,
                        default="data/audio",
                        help="Audio subdirectory")
    parser.add_argument("--annotation_subdir", type=str,
                        default="data/annotations",
                        help="Annotation subdirectory")

    parser.add_argument("--output_dir", type=str,
                       default="data/preprocessed",
                       help="Output directory for preprocessed data")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--n_mels", type=int, default=40, help="Number of Mel filter banks")
    parser.add_argument("--window_size", type=float, default=3.0, help="Window size (seconds)")
    parser.add_argument("--hop_size", type=float, default=1.0, help="Hop size (seconds)")
    parser.add_argument("--event_type", type=str, default="敲击声", help="Event type")

    args = parser.parse_args()

    # Build the full paths
    audio_dir = os.path.join(args.data_dir, args.audio_subdir)
    annotation_dir = os.path.join(args.data_dir, args.annotation_subdir)

    # Check that the directories exist
    if not os.path.exists(audio_dir):
        print(f"Error: audio directory does not exist: {audio_dir}")
        return

    if not os.path.exists(annotation_dir):
        print(f"Error: annotation directory does not exist: {annotation_dir}")
        return

    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%y%m%d%H%M%S"))
    # Run preprocessing
    preprocess_and_save(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        window_size=args.window_size,
        hop_size=args.hop_size,
        event_type=args.event_type
    )


if __name__ == "__main__":
    main()
