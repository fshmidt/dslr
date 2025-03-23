import math

class MathUtils:
    @staticmethod
    def exp(x):
        """Exponential function with clamping to avoid overflow"""
        if x > 250:
            return math.exp(250)
        if x < -250:
            return math.exp(-250)
        return math.exp(x)

    @staticmethod
    def sigmoid(z):
        """Sigmoid activation function"""
        return 1 / (1 + MathUtils.exp(-z))

    @staticmethod
    def dot_product(vec1, vec2):
        """Dot product of two vectors"""
        return sum(a * b for a, b in zip(vec1, vec2))

    @staticmethod
    def add_vectors(vec1, vec2):
        """Add two vectors element-wise"""
        return [a + b for a, b in zip(vec1, vec2)]

    @staticmethod
    def subtract_vectors(vec1, vec2):
        """Subtract two vectors element-wise"""
        return [a - b for a, b in zip(vec1, vec2)]

    @staticmethod
    def scale_vector(vec, scalar):
        """Multiply vector by scalar"""
        return [x * scalar for x in vec]

    @staticmethod
    def min_max_scale(X):
        """Min-max scaling for a list of lists"""
        n_features = len(X[0])
        min_vals = [min(row[i] for row in X) for i in range(n_features)]
        max_vals = [max(row[i] for row in X) for i in range(n_features)]
        range_vals = [max_v - min_v if max_v - min_v != 0 else 1 for max_v, min_v in zip(max_vals, min_vals)]
        X_norm = [[(x - min_v) / r for x, min_v, r in zip(row, min_vals, range_vals)] for row in X]
        return X_norm, min_vals, max_vals

    @staticmethod
    def mean(values):
        """Mean value of a list"""
        return sum(values) / len(values) if values else 0

    @staticmethod
    def log(x, epsilon=1e-5):
        """Logarithm with a small epsilon to avoid log(0)"""
        return math.log(max(x, epsilon))