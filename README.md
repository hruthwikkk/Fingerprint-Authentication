# Fingerprint Authentication System

A Python-based fingerprint authentication system that implements preprocessing, feature extraction, matching, and evaluation components.

## Features

- Image preprocessing with enhancement techniques
- Minutiae-based feature extraction
- Template matching for fingerprint verification
- System performance evaluation with ROC curves and EER

## Project Structure

```
fingerprint_auth/
├── preprocessing.py    # Image enhancement and preprocessing
├── feature_extraction.py   # Minutiae extraction
├── matcher.py         # Template matching
├── evaluation.py      # Performance evaluation
└── main.py           # Main application

data/
├── train/            # Training fingerprint images
├── validate/         # Validation fingerprint images
└── test/            # Test fingerprint images
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place fingerprint images in their respective folders:
   - `data/train/` for enrollment
   - `data/test/` for testing

2. Run the system:
   ```bash
   python main.py
   ```

## Output

The system will:
1. Enroll fingerprints from the training set
2. Test against the test set
3. Display performance metrics:
   - Accuracy
   - ROC curve
   - Score distributions
   - Equal Error Rate (EER)

## File Naming Convention

Fingerprint images should follow the format:
```
YYY_R0_KKK.bmp
```
where:
- `YYY`: Person ID
- `KKK`: Image index