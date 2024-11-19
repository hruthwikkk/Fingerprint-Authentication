import os
import cv2
import numpy as np
from preprocessing import enhance_fingerprint
from feature_extraction import MinutiaeExtractor
from matcher import FingerprintMatcher
from evaluation import SystemEvaluator

def load_dataset(folder_path):
    """
    Load fingerprint images from specified folder.
    Returns dict: {person_id: [image_paths]}
    """
    dataset = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.bmp'):
            person_id = filename.split('_')[0]
            if person_id not in dataset:
                dataset[person_id] = []
            dataset[person_id].append(os.path.join(folder_path, filename))
    return dataset

def main():
    # Initialize components
    extractor = MinutiaeExtractor()
    matcher = FingerprintMatcher(threshold=100)
    evaluator = SystemEvaluator()
    
    # Load datasets
    train_data = load_dataset('data/train')
    test_data = load_dataset('data/test')
    
    # Enrollment phase
    print("Starting enrollment phase...")
    for person_id, image_paths in train_data.items():
        for path in image_paths:
            # Load and preprocess image
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            preprocessed = enhance_fingerprint(image)
            
            # Extract features
            features = extractor.extract_features(preprocessed)
            
            # Enroll person
            matcher.enroll(person_id, features)
    
    # Testing phase
    print("\nStarting testing phase...")
    total_tests = 0
    correct_matches = 0
    
    for true_id, image_paths in test_data.items():
        for path in image_paths:
            # Load and process test image
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            preprocessed = enhance_fingerprint(image)
            features = extractor.extract_features(preprocessed)
            
            # Perform matching
            matched_id, score = matcher.match(features)
            
            # Record results for evaluation
            is_genuine = (true_id == matched_id)
            evaluator.add_comparison(score, is_genuine)
            
            total_tests += 1
            if is_genuine:
                correct_matches += 1
    
    # Calculate and display results
    accuracy = (correct_matches / total_tests) * 100
    print(f"\nSystem Performance:")
    print(f"Total tests: {total_tests}")
    print(f"Correct matches: {correct_matches}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Plot evaluation results
    evaluator.plot_score_distributions()
    auc = evaluator.plot_roc_curve()
    eer = evaluator.calculate_eer()
    
    print(f"\nArea Under ROC Curve (AUC): {auc:.3f}")
    print(f"Equal Error Rate (EER): {eer:.3f}")

if __name__ == "__main__":
    main()