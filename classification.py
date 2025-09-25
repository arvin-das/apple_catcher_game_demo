import random
import mne
import numpy as np
import glob
import json
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing import create_inverse_operator


from constants import *

def check_for_existing_training_data(subject_number):
    folder = f"data/s{subject_number}"
    if len(glob.glob(f"{folder}/*.fif")) == 0:
        return False
    if len(glob.glob(f"{folder}/*.npy")) == 0:
        return False
    return True


def initialize_from_training_data(subject_number):
    folder = f"data/s{str(subject_number).zfill(2)}"
    epochs_all = []
    y_all = []
    X_all = []

    # Get the sorted lists of .fif and .npy files
    fif_files = sorted(glob.glob(f"{folder}/*.fif"))
    npy_files = sorted(glob.glob(f"{folder}/*.npy"))

    # Check if both lists have the same length
    if len(fif_files) != len(npy_files):
        print("Mismatch in the number of .fif and .npy files.")
        return None,None
    
    if len(fif_files) == 0:
        print("\nNOTE: No training data available for this subject. Please select training mode to collect data.")
        return None,None

    for fif_file, npy_file in zip(fif_files, npy_files):
        print(f"Loading {fif_file} and {npy_file}")

        # Load epochs from .fif file
        epochs = mne.read_epochs(fif_file, preload=True, verbose=False)
        # Print number of channels
        print(f"Number of channels: {len(epochs.ch_names)}")
        y = epochs.events[:, -1]
        epochs_all.append(epochs)
        y_all.append(y)

        # Load features from .npy file
        features = np.load(npy_file)
        features = features.reshape(features.shape[0], -1)  # Flatten if needed
        X_all.append(features)

    # Concatenate all epochs and labels
    epochs_all = mne.concatenate_epochs(epochs_all, verbose=False)
    
    y_all = np.concatenate(y_all)
    X_all = np.concatenate(X_all)
    print(f"Feature matrix shape: {X_all.shape}")
    print(f"Label matrix shape: {y_all.shape}")

    # Initialize and train the LDA classifier
    clf = make_pipeline(StandardScaler(),PCA(n_components=0.95),LinearDiscriminantAnalysis())
    clf.fit(X_all, y_all)

    print("Creating inverse operator from training data")
    inverse_operator = create_inverse_operator(epochs_all.info)

    return clf, inverse_operator

def make_giga_classifier(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#,stratify=Y)
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.95), LinearDiscriminantAnalysis())
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def print_results(predictions, y_test):
    print("Results:\n")
    print(f"{'Predictions':<15}{'Actual':<15}")
    print("-" * 30)
    
    for i in range(len(predictions)):
        print(f"{predictions[i][0]:<15}{y_test[i]:<15}")
    
    predictions = np.array(predictions).flatten()
    y_test = np.array(y_test).flatten()
    correct = np.sum(predictions == y_test)
    print(f"Number of correct classifications: {correct}")
    
    accuracy = correct / len(predictions)
    print("\n" + "-" * 30)
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Length of predictions: {len(predictions)}")

def save_results(predictions, y_test, subject_number):
    # Convert to lists for JSON serialization

    predictions = np.array(predictions).flatten()
    y_test = np.array(y_test).flatten()

    score = int(np.sum(predictions == y_test))

    predictions = predictions.tolist()
    y_test = y_test.tolist()

    # Define the file path inside the folder
    folder_path = f"data/s{str(subject_number).zfill(2)}"
    file_path = os.path.join(folder_path, "test_results.json")

    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Compute metrics
    accuracy = accuracy_score(y_test, predictions)
    right_recall = recall_score(y_test, predictions, zero_division=0)  # How well right (1) is predicted
    precision = precision_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Compute left_recall (specificity)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    left_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  # Sensitivity for left (0)

    # Load existing results if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                results = json.load(file)
            except json.JSONDecodeError:
                results = {}  # Handle cases where the file is empty or corrupted
    else:
        results = {}

    # Determine the next test number
    test_number = len(results) + 1
    test_key = f"test_{test_number}"

    # Store new results
    results[test_key] = {
        "score": score,
        "predictions": predictions,
        "y_test": y_test,
        "accuracy": accuracy,
        "right_recall": right_recall, 
        "left_recall": left_recall,  
        "f1": f1,
        "precision": precision
    }

    # Save updated results to file
    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Test {test_number} for subject {subject_number} saved successfully in {file_path}!")
