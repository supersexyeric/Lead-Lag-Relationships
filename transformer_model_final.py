#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SEQUENCE_DATA_PATH = "/home/ubuntu/sequence_data.npz"
MODEL_SAVE_PATH = "/home/ubuntu/transformer_model_final.keras" # Path to save the final trained model
HISTORY_PLOT_PATH = "/home/ubuntu/transformer_training_history_final.png"

# Parameters (Potentially from the report, adjust based on available resources)
# Note: These parameters might need significant reduction (like num_heads, d_model, num_layers, batch_size)
# or training on a subset of data to run in resource-constrained environments.
SEQUENCE_LENGTH = 15 # Adjusted from potential original value due to memory limits
NUM_FEATURES = None # Will be determined from loaded data
HEAD_SIZE = 128     # Size of each attention head
NUM_HEADS = 4       # Number of attention heads (Reduced from potential original)
FF_DIM = 128        # Hidden layer size in feed forward network inside transformer (Reduced)
NUM_TRANSFORMER_BLOCKS = 2 # Number of transformer blocks (Reduced)
MLP_UNITS = [64]    # Units in the final MLP layers (Reduced)
DROPOUT = 0.1
MLP_DROPOUT = 0.2

EPOCHS = 50 # As specified in the report (or adjust as needed)
BATCH_SIZE = 32 # Reduced significantly due to memory limits
VALIDATION_SPLIT = 0.2

# --- Transformer Components ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    """Builds the Transformer model."""
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x) # Output layer for regression

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    print("Transformer Model Summary:")
    model.summary()
    return model

# --- Main Training Logic ---
def train_model():
    """Loads data, trains the Transformer model, and saves it."""
    print(f"Loading sequence data from {SEQUENCE_DATA_PATH}...")
    try:
        data = np.load(SEQUENCE_DATA_PATH)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    except FileNotFoundError:
        print(f"Error: {SEQUENCE_DATA_PATH} not found. Please run prepare_sequence_data_final.py first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if data is sufficient
    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: Training data is empty. Check the preprocessing steps.")
        return
        
    # Determine NUM_FEATURES from the loaded data
    global NUM_FEATURES
    NUM_FEATURES = X_train.shape[2]
    input_shape = (SEQUENCE_LENGTH, NUM_FEATURES)

    # Build the model
    model = build_transformer_model(
        input_shape,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_units=MLP_UNITS,
        dropout=DROPOUT,
        mlp_dropout=MLP_DROPOUT,
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor=\'val_loss\', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor=\'val_loss\', mode=\'min\')

    print("Starting Transformer model training...")
    print("NOTE: Training this model with full data might require significant memory.")
    print(f"Using Sequence Length: {SEQUENCE_LENGTH}, Batch Size: {BATCH_SIZE}")
    
    # Use a subset for demonstration if memory is a concern (as done previously)
    # Modify these lines to use the full dataset if resources allow
    subset_percentage = 0.2 # Using 20% as in the successful LSTM run
    subset_size = int(len(X_train) * subset_percentage)
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    print(f"Training on a subset of {subset_percentage*100}% of the training data ({subset_size} samples) due to potential memory constraints.")

    history = model.fit(X_train_subset, y_train_subset, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=VALIDATION_SPLIT, # Validation split is applied to the subset
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    print(f"Model training finished. Best model (potentially from subset) saved to {MODEL_SAVE_PATH}")

    # Evaluate the best model on the test set
    print("Evaluating model on test data...")
    # Load the best saved model for evaluation
    try:
        best_model = load_model(MODEL_SAVE_PATH)
        test_loss = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (MSE) using best saved model: {test_loss}")
    except Exception as e:
        print(f"Could not load or evaluate the saved model: {e}")
        print("Evaluating with the model state at the end of training (might not be the best).")
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (MSE) using final model state: {test_loss}")


    # Plot training & validation loss values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label=\'Train Loss\')
    plt.plot(history.history["val_loss"], label=\'Validation Loss\')
    plt.title(\'Transformer Model Loss During Training (Subset)\
    plt.ylabel(\'Loss (MSE)\
    plt.xlabel(\'Epoch\')
    plt.legend(loc=\'upper right\')
    plt.grid(True)
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    # Consider memory constraints as noted for the LSTM script
    train_model()

