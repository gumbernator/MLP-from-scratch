# TODO: prevent overshooting due to np.exp()
import numpy as np
from scipy.special import expit as _sigmoid
from scipy.special import softmax as _softmax

# Logistic
def sigmoid(x, deriv = False):
    s = _sigmoid(x)
    if deriv:
        return s * (1 - s)
    else:
        return s

# Hyperbolic tangent
def tanh(x, deriv = False):
    if deriv:
        t = np.tanh(x)
        return 1 - t**2
    else:
        return np.tanh(x)

# Rectified linear
def relu(x, deriv = False):
    r = np.copy(x)
    r[r < 0] = 0
    if deriv:
        r[r > 0] = 1
    return r

def leaky_relu(x, alpha = 0.1, deriv = False):
    r = np.copy(x)
    if deriv:
        r[r > 0] = 1
        r[r < 0] = alpha
    else:
        r[r < 0] *= alpha
    return r

# Probability distibution
def softmax(x, deriv = False):
    s = np.apply_along_axis(_softmax, 1, x)
    if deriv:
        return s * (1 - s)
    else:
        return s