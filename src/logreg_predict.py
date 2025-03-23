import csv
import argparse
from logistic_regression import LogisticRegression

def load_test_data(file_path, means, feature_names):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)
    
    numeric_cols = [headers.index(name) for name in feature_names if name in headers]
    if len(numeric_cols) != len(feature_names):
        raise ValueError(f"Test dataset does not match training features. Expected {feature_names}, got {headers}")
    
    X = []
    indices = []
    
    for row in data:
        features = []
        for idx, mean in zip(numeric_cols, means):
            val = float(row[idx]) if row[idx] else mean
            features.append(val)
        X.append(features)
        indices.append(row[0] if 'Index' in headers else str(len(indices)))
    
    return X, indices

def main():
    parser = argparse.ArgumentParser(description='Predict Hogwarts houses using trained model')
    parser.add_argument('dataset', type=str, help='Path to test CSV file')
    parser.add_argument('model', type=str, help='Path to model parameters file')
    parser.add_argument('--output', type=str, default='houses.csv', help='Output file for predictions')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading model from {args.model}...")
        model = LogisticRegression.load_model(args.model)
        
        print(f"Loading test data from {args.dataset}...")
        X_test, indices = load_test_data(args.dataset, model.means, model.feature_names)
        
        print("Predicting houses...")
        predictions = model.predict(X_test)
        
        print(f"Saving predictions to {args.output}...")
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Hogwarts House'])
            for idx, pred in zip(indices, predictions):
                writer.writerow([idx, pred])
        
        print("Done!")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()