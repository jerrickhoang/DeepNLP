import random
import numpy as np
import math
import timeit
from cs224d.data_utils import *
import matplotlib.pyplot as plt
from scipy.special import expit

#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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
    ### YOUR CODE HERE
    x = (x.T - np.amax(x, axis=1)).T # adjust x for numerical stability.
    expo = np.exp(x)
    res = (expo.T / np.sum(expo, axis=1)).T
    
    ### END YOUR CODE
    
    return res


def sigmoid(x):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################
    MAX = 700
    MIN = 10e-7
    x[x>MAX] = MAX
    res = expit(x)
    res[res<MIN] = MIN
    res[res>1-MIN] = 1-MIN
    return res

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    ###################################################################
    # Compute the gradient for the sigmoid function here. Note that   #
    # for this implementation, the input f should be the sigmoid      #
    # function value of your original input x.                        #
    ###################################################################
    
    ### YOUR CODE HERE
    
    ### END YOUR CODE
    
    return f * ( 1 - f )

def f_sigmoid(x):
    fx = sigmoid(x)
    return fx, sigmoid_grad(fx)

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
    
        ### YOUR CODE HERE: try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        random.setstate(rndstate)
        y = np.copy(x)
        y[ix] += h
        fxp, _ = f(y)
        random.setstate(rndstate)
        y = np.copy(x)
        y[ix] -= h
        fxm, _ = f(y)
        numgrad = (fxp - fxm) / (2*h)

        ### END YOUR CODE
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"


def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector             #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       # 
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    predicted = predicted.reshape((1, len(predicted)))
    dist = predicted.dot(outputVectors.T)
    one_hot = np.zeros(dist.flatten().shape)
    one_hot[target] = 1
    cost = - np.sum(np.log(softmax(dist)) * one_hot)
    # cost = -predicted.dot(outputVectors.T) + np.log(np.sum(np.exp(), axis=1))
    normal_const = np.sum(np.exp(dist), axis=1)
    ddist = -one_hot + (np.exp(dist).T/normal_const).T
    gradPred = ddist.dot(outputVectors)
    gradPred = gradPred.flatten()
    grad = np.outer(ddist, predicted)
    ### END YOUR CODE
    
    return cost, gradPred, grad

def softmaxCostAndGradientDebug(target, params):
    t = 0
    predicted = np.reshape(params[t:t+dimensions[1]], (dimensions[1],))
    t += dimensions[1]
    outputVectors = np.reshape(params[t:t+dimensions[0]*dimensions[1]], (dimensions[0], dimensions[1]))
    cost, gradPred, grad = softmaxCostAndGradient(predicted, target, outputVectors)
    grad_for_gradcheck = np.concatenate((gradPred.flatten(), grad.flatten()))
    return cost, grad_for_gradcheck


def negSamplingCostAndGradient(predicted, target, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    # Did not know why vectorized way didn't work, doing iterative way for now,
    # will take a look later when finished. If training too slow, must look at this.
    ### YOUR CODE HERE
    #MAX = 700
    #MIN = 10e-3
    W = predicted.dot(outputVectors.T)
    #mask = W>MAX
    #W[mask] = MAX
    h = sigmoid(predicted.dot(outputVectors.T))
    neg_samples = [dataset.sampleTokenIdx() for _ in range(K)]
    cost = np.log(h[target])
    for x in neg_samples:
      cost += np.log(h[x])
    #h_negs = h[neg_samples]
    #neg_logs = np.sum(np.log(-h_negs))
    #cost = np.log(h[target]) + neg_logs
    
    #Gradients.
    gradPred = outputVectors[target] * (1 - h[target])
    grad = np.zeros(outputVectors.shape)
    grad[target] += predicted * (1 - h[target])
    for i in range(len(neg_samples)):
        gradPred += outputVectors[neg_samples[i]] * (1 - h[neg_samples[i]])
        grad[neg_samples[i]] += predicted * (1 - h[neg_samples[i]])

    ### END YOUR CODE
    #gradPred[mask] = 0
    #grad[mask] = 0
    return -cost, -gradPred, -grad

def negSamplingCostAndGradientDebug(target, params):
    #dimensions = [5, 4]
    t = 0
    predicted = np.reshape(params[t:t+dimensions[1]], (dimensions[1],))
    t += dimensions[1]
    outputVectors = np.reshape(params[t:t+dimensions[0]*dimensions[1]], (dimensions[0], dimensions[1]))
    cost, gradPred, grad = negSamplingCostAndGradient(predicted, target, outputVectors)
    grad_for_gradcheck = np.concatenate((gradPred.flatten(), grad.flatten()))
    return cost, grad_for_gradcheck


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #         
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of 2*C strings, the context words        #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    inputVectorCurrentWord = inputVectors[tokens[currentWord]]
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for w in contextWords:
        costi, gradInputi, gradOuti = word2vecCostAndGradient(inputVectorCurrentWord, tokens[w], outputVectors)
        cost += costi
        gradIn[tokens[currentWord]] += gradInputi
        gradOut += gradOuti
    ### END YOUR CODE
    
    return cost / 2 / C, gradIn / 2 / C, gradOut / 2 / C

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #         
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################
    
    ### YOUR CODE HERE
    predicted = np.sum(inputVectors, axis=0)
    cost, gradIn, gradOut = word2vecCostAndGradient(predicted, tokens[currentWord], outputVectors)
    gradIn = np.tile(gradIn, (len(inputVectors), 1))
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """    
    return (x.T / np.linalg.norm(x, axis=1)).T


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
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

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 10000

import glob
import os.path as op

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
    ANNEAL_EVERY = 50000
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
        ### YOUR CODE HERE
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.
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
            
        #if iter % ANNEAL_EVERY == 0:
        #    step *= 0.5
    
    return x, loss_history


dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5



# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = normalizeRows(np.random.randn(nWords * 2, dimVectors))
wordVectors0, loss_history = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient), wordVectors, 50.0, 10000, normalizeRows, True)

# just use the output vectors
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:]) / 2.0

print "\n=== For autograder ==="
checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
checkIdx = [tokens[word] for word in checkWords]
checkVecs = wordVectors[checkIdx, :]
print checkVecs
#Visualize the word vectors you trained

_, wordVectors0 = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:]) / 2.0
visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "warm", "enjoyable", "boring", "bad", "garbage", "waste", "disaster", "dumb", "embarrassment", "annoying", "disgusting"]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
covariance = visualizeVecs.T.dot(visualizeVecs)
U,S,V = np.linalg.svd(covariance)
coord = (visualizeVecs - np.mean(visualizeVecs, axis=0)).dot(U[:,0:2]) 

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))
    
#plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
#plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
#plt.show()

np.savetxt('costs.txt', loss_history, fmt='%f')

plt.subplot(2, 1, 1)
plt.plot(loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()

