# Utterance-to-Phoneme-Mapping-using-Sequence-Models

# Overview
This projects implements an Automatic Speech Recognition (ASR) system to map utterances to phoneme sequences. Using a hybrid CNN-RNN architecture with Connectionist Temporal Classification (CTC) loss, the model processes MFCC features from the LibriSpeech dataset (train-clean-100, dev-clean, test-clean) and predicts phoneme transcriptions for 40 phonemes. Leveraging techniques like pyramidal Bi-LSTMs (pBLSTMs), data augmentation, and beam search decoding.

# Objectives

Sequence-to-Sequence Mapping: Convert MFCC features from speech utterances into phoneme sequences using a CNN-RNN model.
Handle Variable-Length Data: Manage asynchronous input-output sequences using padding, packing, and CTC loss.
Optimize Performance: Explore architectures (ResNet-34, pBLSTMs) and decoding strategies (greedy, beam search) to minimize Levenshtein distance on Kaggle.

# Methodology

## Dataset:
Training: LibriSpeech train-clean-100 (MFCC features, phoneme transcripts).
Validation: dev-clean; Testing: test-clean.
Features: 28-dimensional MFCCs per frame; Labels: 40 phonemes + blank (41 total symbols).


## Preprocessing:
Cepstral mean normalization on MFCCs.
Applied TimeMasking and FrequencyMasking for augmentation.
Padded and packed variable-length sequences using PyTorch utilities.


## Model Architecture:
Encoder: ResNet-34 (1D CNNs with SE blocks) for feature extraction, followed by two pBLSTM layers to downsample time resolution and capture temporal dependencies.
Decoder: MLP with log-softmax to output phoneme probabilities.
Total parameters tuned for efficiency (embed_size=856).


## Training:
Used CTC loss to handle time-asynchronous outputs.
Optimized with AdamW (lr=2e-3, weight_decay=1e-3).
Applied CosineAnnealingWarmRestarts scheduler (T_0=5, T_mult=2, eta_min=1e-5).
Trained for 35 epochs with mixed precision on GPU.


## Inference:
Implemented CTC Beam Search decoding (beam_width=4 for validation, 5 for testing).
Computed Levenshtein distance for evaluation.



## Results

Validation: Achieved competitive Levenshtein distance on dev-clean (exact numbers tracked via wandb; best model saved as best.pth).
Testing: Generated phoneme predictions for test-clean, submitted to Kaggle ("submission.csv").
The model effectively handles variable-length sequences and captures long-term dependencies, demonstrating robustness in phoneme transcription.


