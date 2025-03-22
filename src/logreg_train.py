import argparse
import json
import csv
from logistic_regression import LogisticRegression
from math_utils import MathUtils


def load_and_prepare_data(file_path, limit=None):
    """Load and prepare data from CSV"""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)

    if 'Hogwarts House' not in headers:
        raise ValueError("Dataset must contain 'Hogwarts House' column")

    numeric_cols = [i for i, h in enumerate(headers) if h not in ['Index', 'Hogwarts House'] and data[0][i].replace('.', '', 1).lstrip('-').isdigit()]
    if limit:
        data = random.sample(data, min(limit, len(data)))

    X = []
    y = []
    house_idx = headers.index('Hogwarts House')

    for row in data:
        features = []
        for idx in numeric_cols:
            val = float(row[idx]) if row[idx] else MathUtils.mean([float(r[idx]) for r in data if r[idx]])
            features.append(val)
        X.append(features)
        y.append(row[house_idx])
    return X, y, numeric_cols

def moving_average(values, window=3):
    """Compute moving average for smoothing the cost graph"""
    result = []
    for i in range(len(values) - window + 1):
        result.append(sum(values[i:i+window]) / window)
    return result

def visualize_training(costs_dict, save_path=None):
    """Visualize training progress"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for class_val, costs in costs_dict.items():
            smooth_costs = moving_average(costs)
            plt.plot(smooth_costs, label=f'Class: {class_val}')
        plt.title('Cost Function during Training')
        plt.xlabel('Iterations (x50)')
        plt.ylabel('Cost')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except ImportError:
        print("Matplotlib not available, skipping visualization")

def main():
    parser = argparse.ArgumentParser(description='Train logistic regression classifier')
    parser.add_argument('dataset', type=str, help='Path to training dataset CSV file')
    parser.add_argument('--output', type=str, default='model.json', help='Output file for model parameters')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--iter', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('--visualize', action='store_true', help='Visualize training progress')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for faster training')
    args = parser.parse_args()

    try:
        print(f"Loading data from {args.dataset}...")
        X, y, _ = load_and_prepare_data(args.dataset, limit=args.limit)
        print(f"Training on {len(X)} samples with {len(X[0])} features...")

        model = LogisticRegression(learning_rate=args.lr, max_iter=args.iter)
        costs_dict = model.fit(X, y)

        y_pred = model.predict(X)
        accuracy = sum(1 for pred, true in zip(y_pred, y) if pred == true) / len(y)
        print(f"Training accuracy: {accuracy:.4f}")

        model.save_model(args.output)
        print(f"Model saved to {args.output}")

        if args.visualize:
            visualize_training(costs_dict)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()