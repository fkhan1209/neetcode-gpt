import numpy as np
from numpy.typing import NDArray
import math


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        res = 1 / (1 + ((math.e)**(-1 * z)))
        return np.round(res, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        res = np.maximum (0, z)
        return res
