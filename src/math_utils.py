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
    def matrix_vector_mult(matrix, vector):
        """Matrix-vector multiplication"""
        return [MathUtils.dot_product(row, vector) for row in matrix]

    @staticmethod
    def transpose(matrix):
        """Matrix transposition"""
        return list(map(list, zip(*matrix)))

    @staticmethod
    def mean(values):
        """Mean value of a list"""
        return sum(values) / len(values) if values else 0

    @staticmethod
    def std(values, mean_val):
        """Standard deviation of a list"""
        if not values:
            return 1
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return math.sqrt(variance) if variance > 0 else 1

    @staticmethod
    def standardize(data, mean=None, std=None):
        """Standardize data: (x - mean) / std"""
        if mean is None:
            mean = [MathUtils.mean(col) for col in MathUtils.transpose(data)]
        if std is None:
            std = [MathUtils.std(col, m) for col, m in zip(MathUtils.transpose(data), mean)]
        result = [[(x - m) / s for x, m, s in zip(row, mean, std)] for row in data]
        return result, mean, std

    @staticmethod
    def log(x, epsilon=1e-5):
        """Logarithm with a small epsilon to avoid log(0)"""
        return math.log(max(x, epsilon))