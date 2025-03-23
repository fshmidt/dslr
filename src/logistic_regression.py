import json
from math_utils import MathUtils

class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=1000, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.weights = {}
        self.classes = None
        self.features = None
        self.min_vals = None
        self.max_vals = None
        self.means = None
        self.feature_names = None
    
    def compute_cost(self, X, y, weights):
        m = len(X)
        h = [MathUtils.sigmoid(MathUtils.dot_product(x, weights)) for x in X]
        cost = -(1/m) * sum(yi * MathUtils.log(hi) + (1 - yi) * MathUtils.log(1 - hi) for yi, hi in zip(y, h))
        return cost
    
    def gradient_descent(self, X, y, weights):
        m = len(X)
        h = [MathUtils.sigmoid(MathUtils.dot_product(x, weights)) for x in X]
        gradient = [0] * len(weights)
        for j in range(len(weights)):
            gradient[j] = (1/m) * sum((hi - yi) * xi[j] for hi, yi, xi in zip(h, y, X))
        return gradient
    
    def fit_one_vs_all(self, X, y, class_val):
        n = len(X[0])
        X_with_bias = [[1] + row for row in X]
        y_binary = [1 if yi == class_val else 0 for yi in y]
        weights = [0] * (n + 1)
        costs = []
        for i in range(self.max_iter):
            gradient = self.gradient_descent(X_with_bias, y_binary, weights)
            weights = MathUtils.subtract_vectors(weights, MathUtils.scale_vector(gradient, self.learning_rate))
            if i % 50 == 0:
                cost = self.compute_cost(X_with_bias, y_binary, weights)
                costs.append(cost)
                if len(costs) > 1 and abs(costs[-1] - costs[-2]) < self.epsilon:
                    break
        return weights, costs
    
    def fit(self, X_train, y_train, X_val, y_val):
        X_train_norm, self.min_vals, self.max_vals = MathUtils.min_max_scale(X_train)\

        self.classes = list(set(y_train))
        self.features = len(X_train[0])

        costs_dict = {}
        for class_val in self.classes:
            weights, costs = self.fit_one_vs_all(X_train_norm, y_train, class_val)
            self.weights[class_val] = weights
            costs_dict[class_val] = costs

        val_predictions = self.predict(X_val)
        val_accuracy = sum(1 for pred, true in zip(val_predictions, y_val) if pred == true) / len(y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        return costs_dict
    
    def predict_proba(self, X):
        X_norm = [[(x - min_v) / (max_v - min_v if max_v - min_v != 0 else 1) 
                   for x, min_v, max_v in zip(row, self.min_vals, self.max_vals)] 
                  for row in X]
        m = len(X_norm)
        X_with_bias = [[1] + row for row in X_norm]
        probas = {}
        for class_val, weights in self.weights.items():
            probas[class_val] = [MathUtils.sigmoid(MathUtils.dot_product(x, weights)) for x in X_with_bias]
        return probas
    
    def predict(self, X):
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
        model_data = {
            'weights': self.weights,
            'min_vals': self.min_vals,
            'max_vals': self.max_vals,
            'means': self.means,
            'classes': self.classes,
            'feature_names': self.feature_names
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)
        model = cls()
        model.weights = model_data['weights']
        model.min_vals = model_data['min_vals']
        model.max_vals = model_data['max_vals']
        model.means = model_data['means']
        model.classes = model_data['classes']
        model.feature_names = model_data['feature_names']
        model.features = len(model.min_vals)
        return model