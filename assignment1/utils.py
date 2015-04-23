import numpy as np
import random
from scipy.special import expit
def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################
    x = (x.T - np.amax(x, axis=1)).T # adjust x for numerical stability.
    expo = np.exp(x)
    res = (expo.T / np.sum(expo, axis=1)).T
    return res


def sigmoid(x):
    MAX = 700
    MIN = 10e-7
    x[x>MAX] = MAX
    res = expit(x)
    res[res<MIN] = MIN
    res[res>1-MIN] = 1-MIN
    return res


def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
    
        random.setstate(rndstate)
        y = np.copy(x)
        y[ix] += h
        fxp, _ = f(y)
        random.setstate(rndstate)
        y = np.copy(x)
        y[ix] -= h
        fxm, _ = f(y)
        numgrad = (fxp - fxm) / (2*h)

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

