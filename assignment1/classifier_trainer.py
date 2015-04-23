import numpy as np
from models import *
from utils import gradcheck_naive 
import glob
import os.path as op
# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 10000


def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        return st, np.load("saved_params_%d.npy" % st)
    else:
        return st, None
    
def save_params(iter, params):
    np.save("saved_params_%d.npy" % iter, params)


def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False):
    """ Stochastic Gradient Descent """
    ###################################################################
    # Implement the stochastic gradient descent method in this        #
    # function.                                                       #
    # Inputs:                                                         #
    #   - f: the function to optimize, it should take a single        #
    #        argument and yield two outputs, a cost and the gradient  #
    #        with respect to the arguments                            #
    #   - x0: the initial point to start SGD from                     #
    #   - step: the step size for SGD                                 #
    #   - iterations: total iterations to run SGD for                 #
    #   - postprocessing: postprocessing function for the parameters  #
    #        if necessary. In the case of word2vec we will need to    #
    #        normalize the word vectors to have unit length.          #
    # Output:                                                         #
    #   - x: the parameter value after SGD finishes                   #
    ###################################################################
    
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    loss_history = []
    
    if useSaved:
        start_iter, oldx = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
    else:
        start_iter = 0
    x = x0
    if not postprocessing:
        postprocessing = lambda x: x
    
    expcost = None
    # FOR RMS PROP
    decay_rate = 0.99
    momentum = 0.9
    rmsgrad_cache = 0
    rmsstep_cache = 0
    for iter in xrange(start_iter + 1, iterations + 1):
        cost, grad = f(x)
        rmsstep_cache = rmsstep_cache * momentum - step * grad 
        x +=  rmsstep_cache
        #gradcheck_naive(f, x)
        loss_history.append(cost)
        if iter % 1000 == 0:
            print cost
        postprocessing(x)
        ### END YOUR CODE
        
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
            
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    
    return x, loss_history

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """    
    #return (x.T / np.linalg.norm(x, axis=1)).T
    x2 = x*x
    if len(np.shape(x)) <= 1:
	return np.sqrt(x2/np.sum(x2))
    return np.sqrt(x2/(np.sum(x2,axis=1)[:, np.newaxis]))


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 20
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        context = dataset.getRandomContext(C)
        c, gin, gout = word2vecModel(context[C], C, context[:C] + context[C+1:], tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize
        grad[:N/2, :] += gin / batchsize
        grad[N/2:, :] += gout / batchsize
        
    return cost, grad

# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return [tokens[random.randint(0,4)] for i in xrange(2*C+1)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext
#######################
##Grad check purpose - should comment out when done
dimensions = [5, 10]
target = random.randint(0,dimensions[0]-1)
params = np.random.randn(dimensions[0]*dimensions[1] + dimensions[1], )


random.seed(31415)
np.random.seed(9265)
dummy_vectors = normalizeRows(np.random.randn(10,3))
dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
print "==== Gradient check for skip-gram ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
print "\n==== Gradient check for CBOW      ===="
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

print "\n=== For autograder ==="
print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)

