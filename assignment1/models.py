from utils import *
import numpy as np

# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return [tokens[random.randint(0,4)] for i in xrange(2*C+1)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext


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
    ###################################################################

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
    # size.
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    ###################################################################
    
    # Did not know why vectorized way didn't work, doing iterative way for now,
    # will take a look later when finished. If training too slow, must look at this.
    W = predicted.dot(outputVectors.T)
    h = sigmoid(predicted.dot(outputVectors.T))
    neg_samples = [dataset.sampleTokenIdx() for _ in range(K)]
    cost = np.log(h[target])
    cost += np.sum(np.log(h[neg_samples]))
    #for x in neg_samples:
    #  cost += np.log(h[x])
    
    #Gradients.
    gradPred = outputVectors[target] * (1 - h[target])
    grad = np.zeros(outputVectors.shape)
    grad[target] += predicted * (1 - h[target])
    for i in range(len(neg_samples)):
        gradPred += outputVectors[neg_samples[i]] * (1 - h[neg_samples[i]])
        grad[neg_samples[i]] += predicted * (1 - h[neg_samples[i]])

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


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    predicted = np.sum(inputVectors, axis=0)
    cost, gradIn, gradOut = word2vecCostAndGradient(predicted, tokens[currentWord], outputVectors)
    gradIn = np.tile(gradIn, (len(inputVectors), 1))
    
    return cost, gradIn, gradOut

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
    ###################################################################
    
    inputVectorCurrentWord = inputVectors[tokens[currentWord]]
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for w in contextWords:
        costi, gradInputi, gradOuti = word2vecCostAndGradient(inputVectorCurrentWord, tokens[w], outputVectors)
        cost += costi
        gradIn[tokens[currentWord]] += gradInputi
        gradOut += gradOuti
    
    return cost / 2 / C, gradIn / 2 / C, gradOut / 2 / C

