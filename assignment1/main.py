from cs224d.data_utils import *
from classifier_trainer import *
import matplotlib.pyplot as plt

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
print wordVectors.shape
wordVectors0, loss_history = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient), wordVectors, 50.0, 20000, normalizeRows, True)

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
    
plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
plt.savefig('words-viz.png')
plt.show()

np.savetxt('costs.txt', loss_history, fmt='%f')

plt.subplot(2, 1, 1)
plt.plot(loss_history)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('loss-history.png')

plt.show()

