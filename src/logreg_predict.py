import csv
from logistic_regression import LogisticRegression
import json
import argparse
from math_utils import MathUtils

def load_test_data(file_path):
    """Load test data from CSV"""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)

    numeric_cols = [i for i, h in enumerate(headers) if h not in ['Index', 'Hogwarts House'] and data[0][i].replace('.', '', 1).lstrip('-').isdigit()]
    
    X = []
    indices = []
    
    for row in data:
        features = []
        for idx in numeric_cols:
            val = float(row[idx]) if row[idx] else 0  # Replace missing values with 0
            features.append(val)
        X.append(features)
        indices.append(row[0] if 'Index' in headers else str(len(indices)))
    
    return X, indices

def save_predictions(indices, predictions, filename='houses.csv'):
    """Save predictions to CSV file in required format"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Hogwarts House'])
        for idx, pred in zip(indices, predictions):
            writer.writerow([idx, pred])

def main():
    parser = argparse.ArgumentParser(description='Predict Hogwarts houses using logistic regression')
    parser.add_argument('dataset', type=str, help='Path to test dataset CSV file')
    parser.add_argument('weights', type=str, default='model.json', help='Path to file with trained weights')
    args = parser.parse_args()

    try:
        print(f"Loading test data from {args.dataset}...")
        X, indices = load_test_data(args.dataset)
        print(f"Predicting for {len(X)} samples with {len(X[0])} features...")

        print(f"Loading model from {args.weights}...")
        model = LogisticRegression.load_model(args.weights)

        predictions = model.predict(X)
        save_predictions(indices, predictions)
        print("Predictions saved to houses.csv")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()