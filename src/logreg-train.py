import argparse
import json
import csv
from math_utils import MathUtils
import random
from multiprocessing import Pool

def fit_class_wrapper(args):
    """Function for parallel training of one class"""
    class_val, X_data, y_data, max_iter, learning_rate, epsilon = args
    lr_temp = LogisticRegression(learning_rate=learning_rate, max_iter=max_iter, epsilon=epsilon)
    return class_val, lr_temp.fit_one_vs_all(X_data, y_data, class_val)

class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.weights = {}
        self.classes = None
        self.features = None
        self.mean = None
        self.std = None

    def compute_cost(self, X, y, weights):
        """Compute the logistic regression cost"""
        m = len(X)
        h = [MathUtils.sigmoid(MathUtils.dot_product(x, weights)) for x in X]
        cost = -(1/m) * sum(yi * MathUtils.log(hi) + (1-yi) * MathUtils.log(1-hi) for yi, hi in zip(y, h))
        return cost

    def gradient_descent(self, X, y, weights, batch_size=32):
        """Gradient descent with mini-batches"""
        m = len(X)
        gradient = [0] * len(weights)
        
        indices = list(range(m))
        random.shuffle(indices)
        
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch_indices = indices[start:end]
            batch_X = [X[i] for i in batch_indices]
            batch_y = [y[i] for i in batch_indices]
            
            h = [MathUtils.sigmoid(MathUtils.dot_product(x, weights)) for x in batch_X]
            errors = [(hi - yi) for hi, yi in zip(h, batch_y)]
            
            X_T = MathUtils.transpose(batch_X)
            for j in range(len(weights)):
                gradient[j] += (1/len(batch_X)) * sum(err * xj for err, xj in zip(errors, X_T[j]))
        
        gradient = [g * (batch_size/m) for g in gradient]
        return gradient

    def fit_one_vs_all(self, X, y, class_val):
        """Train logistic regression for one class vs all others"""
        m = len(X)
        n = len(X[0])
        X_with_bias = [[1] + row for row in X]
        y_binary = [1 if yi == class_val else 0 for yi in y]
        weights = [0] * (n + 1)

        costs = []
        for i in range(self.max_iter):
            gradient = self.gradient_descent(X_with_bias, y_binary, weights, batch_size=32)
            weights = [w - self.learning_rate * g for w, g in zip(weights, gradient)]
            if i % 50 == 0:
                cost = self.compute_cost(X_with_bias, y_binary, weights)
                costs.append(cost)
                if len(costs) > 1 and abs(costs[-1] - costs[-2]) < self.epsilon:
                    break
        return weights, costs

    def fit(self, X, y):
        """Train model for all classes using one-vs-all approach"""
        X = [list(map(float, row)) for row in X]
        y = list(y)
        X_norm, self.mean, self.std = MathUtils.standardize(X)
        self.classes = list(set(y))
        self.features = len(X[0])
        costs_dict = {}

        tasks = [(class_val, X_norm, y, self.max_iter, self.learning_rate, self.epsilon) for class_val in self.classes]
        with Pool() as pool:
            results = pool.map(fit_class_wrapper, tasks)

        for class_val, (weights, costs) in results:
            self.weights[class_val] = weights
            costs_dict[class_val] = costs
        return costs_dict

    def predict_proba(self, X):
        """Predict probability for each class"""
        X = [list(map(float, row)) for row in X]
        X_norm = [[(x - m) / s for x, m, s in zip(row, self.mean, self.std)] for row in X]
        
        m = len(X_norm)
        X_with_bias = [[1] + row for row in X_norm]
        
        probas = {}
        for class_val, weights in self.weights.items():
            probas[class_val] = [MathUtils.sigmoid(MathUtils.dot_product(x, weights)) for x in X_with_bias]
        return probas

    def predict(self, X):
        """Predict class with highest probability"""
        probas = self.predict_proba(X)
        
        m = len(X)
        predictions = []
        
        for i in range(m):
            max_prob = -1
            max_class = None
            for class_val, probs in probas.items():
                if probs[i] > max_prob:
                    max_prob = probs[i]
                    max_class = class_val
            predictions.append(max_class)
        
        return predictions

    def save_model(self, filename):
        """Save model parameters to file"""
        model_data = {
            'weights': {str(k): v for k, v in self.weights.items()},
            'mean': self.mean,
            'std': self.std,
            'classes': self.classes
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

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