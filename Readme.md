# Apple Catcher Game – EEG-Driven Interaction

This project uses Lab Streaming Layer (LSL) and several Python libraries to process EEG data and interact with a simple game using MNE, Pygame, and more.

## Requirements

- Python 3.9+ (ideally using a virtual environment)
- LSL native library (`liblsl`)
- Packages listed in `requirements.txt`

---

## Setup Instructions

### macOS

1. Install Homebrew (if not already):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install the LabStreamingLayer native library:

   ```bash
   brew tap labstreaminglayer/tap
   brew install lsl
   ```

3. Set up Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```


### Ubuntu

1. Install dependencies:

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip
   ```

2. Download and install the native LSL library:

   - Go to https://github.com/sccn/liblsl/releases
   - Download the appropriate `.deb` file (e.g., `liblsl-1.16.2-focal_amd64.deb`)
   - Install it:

     ```bash
     sudo apt install ./liblsl-1.16.2-focal_amd64.deb
     sudo apt install --fix-broken --yes
     ```

3. Set up Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Project Structure

Here's a breakdown of the key files in the project and their roles:

### `apple_catcher_game.py`

Main script. Runs the EEG-controlled Apple Catcher game. Handles:
- Game logic and user interface using `pygame`
- Connecting to LSL EEG stream
- Real-time EEG data collection and classification
- Storing results and features

### `classification.py`

Provides classification tools:
- Loads training data and labels
- Trains a classifier using `PCA` + `LDA`
- Predicts new labels in real time
- Saves and prints evaluation metrics

### `constants.py`

Defines constants used throughout the project:
- Game settings (screen size, timing, etc.)
- EEG stream info and preprocessing parameters
- Image paths and color codes
- Channel names for different EEG headsets

### `preprocessing.py`

Handles EEG data processing using `mne`:
- Filters and references EEG signals
- Creates epochs from sample windows
- Computes inverse operator for source reconstruction
- Extracts spatio-spectral features from each epoch

### `data_collection.py`

Connects to LSL stream and handles real-time data:
- Creates LSL inlets
- Retrieves and formats sample buffers
- Converts raw data into MNE-compatible format
- Includes optional support for loading `.mat` datasets
