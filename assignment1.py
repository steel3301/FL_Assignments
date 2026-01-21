"""
Assignment 1: Federated Learning Simulation

This program simulates a basic Federated Learning system.
Multiple clients train local models on their own data without
sharing raw datasets. A central server aggregates the models
using Federated Averaging (FedAvg).

Dataset: MNIST (automatically downloaded)
Model: Simple Logistic Regression-style Neural Network
Output: Saved to 'assignment1-output.txt'
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
import os

# -----------------------------
# Output logging setup
# -----------------------------
OUTPUT_FILE = "assignment1-output.txt"

# Clear previous output file if it exists
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

def log(message):
    print(message)
    with open(OUTPUT_FILE, "a") as f:
        f.write(message + "\n")

# -----------------------------
# Configuration
# -----------------------------
NUM_CLIENTS = 5
LOCAL_EPOCHS = 1
ROUNDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

log("Federated Learning Assignment 1")
log("--------------------------------")
log(f"Number of clients: {NUM_CLIENTS}")
log(f"Federated rounds: {ROUNDS}")
log("")

# -----------------------------
# Load and preprocess dataset
# -----------------------------
log("Downloading and preparing MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# Split data among clients
client_data_size = len(x_train) // NUM_CLIENTS
client_datasets = []

for i in range(NUM_CLIENTS):
    start = i * client_data_size
    end = start + client_data_size
    client_datasets.append((x_train[start:end], y_train[start:end]))

log("Dataset split among clients successfully.\n")

# -----------------------------
# Model definition
# -----------------------------
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=SGD(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -----------------------------
# Federated Averaging function
# -----------------------------
def federated_average(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

# -----------------------------
# Initialize global model
# -----------------------------
global_model = create_model()
global_weights = global_model.get_weights()

log("Global model initialized.\n")

# -----------------------------
# Federated Learning process
# -----------------------------
for round_num in range(1, ROUNDS + 1):
    log(f"--- Federated Learning Round {round_num} ---")

    local_weights = []

    for client_id, (client_x, client_y) in enumerate(client_datasets):
        log(f"Client {client_id + 1} training on local data...")

        local_model = create_model()
        local_model.set_weights(global_weights)

        local_model.fit(
            client_x,
            client_y,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        local_weights.append(local_model.get_weights())

    # Aggregate client models
    global_weights = federated_average(local_weights)
    global_model.set_weights(global_weights)

    # Evaluate global model
    loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    log(f"Global Model Accuracy after round {round_num}: {accuracy:.4f}\n")

# -----------------------------
# Final Evaluation
# -----------------------------
log("Final Global Model Evaluation")
loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
log(f"Final Test Accuracy: {accuracy:.4f}")

log("\nProgram execution completed.")
log(f"Output saved to '{OUTPUT_FILE}'")
