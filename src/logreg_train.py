import pandas as pd
import matplotlib.pyplot as plt
import argparse
from math_utils import MathUtils
from logistic_regression import LogisticRegression

def load_and_prepare_data(file_path, means=None):
    data = pd.read_csv(file_path)
    if 'Hogwarts House' not in data.columns:
        raise ValueError("Dataset must contain 'Hogwarts House' column")
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Index' in numeric_features:
        numeric_features.remove('Index')
    data = data.dropna(subset=numeric_features)
    X = [list(row) for row in data[numeric_features].values]
    y = list(data['Hogwarts House'].values)
    if means is None:
        means = [MathUtils.mean([row[i] for row in X]) for i in range(len(X[0]))]
    else:
        X = [[means[i] if pd.isna(x) else x for i, x in enumerate(row)] for row in X]
    return X, y, numeric_features, means

def visualize_training(costs_dict):
    plt.figure(figsize=(10, 6))
    for class_val, costs in costs_dict.items():
        plt.plot(costs, label=f'Class: {class_val}')
    plt.title('Cost Function during Training')
    plt.xlabel('Iterations (x50)')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

def train_test_split(X, y, test_size=0.2, random_state=42):
    import random
    random.seed(random_state)
    data = list(zip(X, y))
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    X_train, y_train = zip(*data[:split_idx])
    X_val, y_val = zip(*data[split_idx:])
    return list(X_train), list(y_train), list(X_val), list(y_val)

def main():
    parser = argparse.ArgumentParser(description='Train logistic regression classifier for Hogwarts houses')
    parser.add_argument('dataset', type=str, help='Path to training CSV file')
    parser.add_argument('--output', type=str, default='model.json', help='Output file for model parameters')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--iter', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('--visualize', action='store_true', help='Visualize training progress')
    parser.add_argument('--gd', type=str, choices=['standard', 'batch', 'stochastic'], default='standard',help='Type of gradient descent: standard/batch/stochastic')

    args = parser.parse_args()
    
    try:
        print(f"Loading data from {args.dataset}...")
        X, y, feature_names, means = load_and_prepare_data(args.dataset)

        X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples with {len(X_train[0])} features...")

        model = LogisticRegression(learning_rate=args.lr, max_iter=args.iter, gd=args.gd)
        model.means = means
        model.feature_names = feature_names
        costs_dict = model.fit(X_train, y_train, X_val, y_val)
        
        model.save_model(args.output)
        print(f"Model saved to {args.output}")
        
        if args.visualize:
            visualize_training(costs_dict)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()