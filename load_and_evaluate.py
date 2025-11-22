"""
Load a saved model and run generalization evaluation without retraining.

Usage:
    python load_and_evaluate.py
"""

import pickle
from pathlib import Path
import tensorflow as tf
from evaluate_generalization import run_evaluation


def load_model_and_encoder(
    model_path="saved_models/math_model.keras",
    encoder_path="saved_models/character_label_encoder.pkl"
):
    """Load saved model and character encoder."""

    model_path = Path(model_path)
    encoder_path = Path(encoder_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run fit_model.py first to train and save a model."
        )

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Encoder not found at {encoder_path}. "
            f"Run fit_model.py first to train and save a model."
        )

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading character encoder from {encoder_path}...")
    with open(encoder_path, 'rb') as f:
        character_label_encoder = pickle.load(f)

    print("Model and encoder loaded successfully!")
    print(f"Model summary:")
    model.summary()

    return model, character_label_encoder


def main():
    """Load saved model and run evaluation."""

    try:
        model, character_label_encoder = load_model_and_encoder()

        print("\n" + "="*80)
        print("RUNNING GENERALIZATION EVALUATION ON LOADED MODEL")
        print("="*80)

        run_evaluation(model, character_label_encoder)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
