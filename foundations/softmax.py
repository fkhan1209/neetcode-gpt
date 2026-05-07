import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        new_arr = np.exp(z - np.max(z))
        return np.round((new_arr/ new_arr.sum()), 4) 
