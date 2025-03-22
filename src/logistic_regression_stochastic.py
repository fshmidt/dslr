import json
from math_utils import MathUtils
import random
from multiprocessing import Pool

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

    @classmethod
    def load_model(cls, filename):
        """Load model parameters from file"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        model = cls()
        model.weights = {k: v for k, v in model_data['weights'].items()}
        model.mean = model_data['mean']
        model.std = model_data['std']
        model.classes = model_data['classes']
        model.features = len(model.mean)
        return model
        
    def compute_cost(self, X, y, weights):
        """Compute the logistic regression cost"""
        m = len(X)
        h = [MathUtils.sigmoid(MathUtils.dot_product(x, weights)) for x in X]
        cost = -(1/m) * sum(yi * MathUtils.log(hi) + (1-yi) * MathUtils.log(1-hi) for yi, hi in zip(y, h))
        return cost


    def stochastic_gradient_descent(self, X, y, weights):
        # Initialize gradient
        n_features = len(weights)
        gradient = [0] * n_features

        # Get a single random sample
        idx = random.randint(0, len(X) - 1)
        x_i = X[idx]
        y_i = y[idx]

        h = MathUtils.sigmoid(MathUtils.dot_product(x_i, weights))
        error = h - y_i

        for j in range(n_features):
            gradient[j] = error * x_i[j]
        
        return gradient


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
            gradient = self.stochastic_gradient_descent(X_with_bias, y_binary, weights)
            weights = [w - self.learning_rate * g for w, g in zip(weights, gradient)]
            
            # Compute cost periodically to check convergence
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

def fit_class_wrapper(args):
    """Function for parallel training of one class"""
    class_val, X_data, y_data, max_iter, learning_rate, epsilon = args
    lr_temp = LogisticRegression(learning_rate=learning_rate, max_iter=max_iter, epsilon=epsilon)
    return class_val, lr_temp.fit_one_vs_all(X_data, y_data, class_val)