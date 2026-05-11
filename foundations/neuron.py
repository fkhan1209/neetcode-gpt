import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.dot(x, w) + b
        ans = -1 
        if activation == "sigmoid": 
            ans = 1/ (1 + (math.exp(-z)))
        else:
            ans = max(0, z)
        return np.round(float(ans), 5)
