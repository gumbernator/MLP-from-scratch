# Predicted : _y
# Actual : y
# _y, y must be ndarrays
import numpy as np

# Mean Squared Error (Regression)
def mse(_y, y, deriv = False):
    if deriv:
        return _y - y
    else:
        return (_y - y)**2 / 2

# Mean Absolute Error
def mae(_y, y, deriv = False):
    if deriv:
        diff = _y - y
        diff[diff > 0] = 1
        diff[diff < 0] = -1
        return diff
    else:
        return abs(_y - y)

# Classification Loss
def cross_entropy(_y, y, deriv = False):
    if deriv:
        return _y - y
    else:
        return -y * np.log(_y + 1e-5)
    