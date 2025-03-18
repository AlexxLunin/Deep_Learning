# Speech Command Recognition with CRNN

This project implements a speech command recognition system using a Convolutional Recurrent Neural Network (CRNN). The model is trained on the SPEECHCOMMANDS dataset from torchaudio and is capable of classifying various voice commands.

## Project Overview

The project employs a hybrid CRNN architecture that combines the strengths of Convolutional Neural Networks (CNN) for extracting spatial features from spectrograms and Recurrent Neural Networks (RNN) for modeling temporal dependencies in audio data.

## Features

- Automatic downloading and processing of the SPEECHCOMMANDS dataset
- Conversion of audio signals to mel-spectrograms
- CRNN architecture using CNN and GRU layers
- Training visualization with TensorBoard
- Comprehensive model evaluation on the test dataset

## Model Architecture

The CRNN model consists of the following components:
- Adaptive pooling for input size normalization
- Two convolutional blocks with batch normalization and max-pooling
- Two GRU (Gated Recurrent Unit) layers for sequence processing
- A fully connected layer for classification

## Requirements

```
python >= 3.6
torch
torchaudio
tensorboard
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/speech-command-recognition.git
cd speech-command-recognition
```

2. Install dependencies:
```bash
pip install torch torchaudio tensorboard
```

## Usage

Run the main script to train and evaluate the model:

```bash
python Model.py
```

The script will automatically:
1. Download the SPEECHCOMMANDS dataset (if not already downloaded)
2. Preprocess audio data and create mel-spectrograms
3. Split the data into training, validation, and test sets
4. Train the CRNN model
5. Evaluate the model's performance on the test set
6. Save the trained model to `model.pth`

## Monitoring Training Progress

During training, loss and accuracy metrics will be logged to TensorBoard. To visualize the training progress, run:

```bash
tensorboard --logdir=runs
```

Then open the provided URL in your browser (typically http://localhost:6006).

## Project Structure

- `SPEECHCOMMANDSDataset`: Class for preparing and preprocessing audio data
- `CRNN`: Implementation of the Convolutional Recurrent Neural Network model
- Main code block for training and evaluating the model

## Data Processing Pipeline

1. Audio files are loaded using torchaudio
2. Audio signals are normalized to a fixed length (16000 samples)
3. Mel-spectrograms are generated using the following parameters:
   - Sample rate: 16000 Hz
   - FFT window size: 512
   - Hop length: 160
   - Number of mel filters: 64
4. Spectrograms are normalized to a fixed size (128 time steps)

## Performance

After training for 20 epochs, the model achieves significant accuracy on the test dataset. The exact metrics depend on the dataset size and composition, as well as the model hyperparameters.

## Future Improvements

- Extending the model to work with arbitrary audio data
- Optimizing hyperparameters through automated search
- Implementing data augmentation techniques to improve generalization
- Model quantization for more efficient deployment
- Adding a web interface for model testing

## License

MIT
