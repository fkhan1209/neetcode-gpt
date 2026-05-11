import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Create a duplicate of initial_weights as weights
        weights = initial_weights
        # For each iteration in the range of num_iterations
        for i in range(num_iterations):
            # predictions is calculated with get_model_prediction class method, using X and the weights
            preds = self.get_model_prediction(X, weights)
            # For each weight
            for j in range(len(weights)): 
                # Calculate the gradient with get_derivative class method, using predictions (the predicted matrix), 
                # Y (the target values), len(X) <-- which is the same as N in the get_derivative class method, 
                # X (the feature matrix), and j (the specific index of the weight you are calculating gradient for)    
                grad = self.get_derivative(preds, Y, len(X), X, j)
                # Update weight with the formula: w = w - (learning_rate * gradient)
                weights[j] -= self.learning_rate * grad 
        #Return the weights rounded to 5 figures
        return np.round(weights, 5)


