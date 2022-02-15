# Solution set for CS 155 Set 6, 2017
# Authors: Suraj Nair, Sid Murching

from keras.models import Sequential
from P3CHelpers import *
import sys

# Additional imports
from keras.layers.core import Dense

def get_word_repr(word_to_index, word):
    """
    Returns one-hot-encoded feature representation of the specified word given
    a dictionary mapping words to their one-hot-encoded index.

    Arguments:
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        word:          String containing word whose feature representation we wish to compute.

    Returns:
        feature_representation:     Feature representation of the passed-in word.
    """
    unique_words = word_to_index.keys()
    # Return a vector that's zero everywhere besides the index corresponding to <word>
    feature_representation = np.zeros(len(unique_words))
    feature_representation[word_to_index[word]] = 1
    return feature_representation    

def generate_traindata(word_list, word_to_index, window_size=4):
    """
    Generates training data for Skipgram model.

    Arguments:
        word_list:     Sequential list of words (strings).
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        window_size:   Size of Skipgram window.
                       (use the default value when running your code).

    Returns:
        (trainX, trainY):     A pair of matrices (trainX, trainY) containing training 
                              points (one-hot-encoded vectors representing individual words) and 
                              their corresponding labels (also one-hot-encoded vectors representing words).

                              For each index i, trainX[i] should correspond to a word in
                              <word_list>, and trainY[i] should correspond to one of the words within
                              a window of size <window_size> of trainX[i].
    """
    trainX = []
    trainY = []
    # TODO: Implement this function, populating trainX and trainY
    trainX = []
    trainY = []
    for i in range(len(trainX)):
        window_front = word_list[i:i+window_size]
        window_back = np.flip(word_list[i-window_size+1:i+1])

        trainX_vec = np.zeros(len(word_to_index))
        trainX_vec[word_to_index[window_front[0]]] = 1
        trainX.append(trainX_vec)
        trainX.append(trainX_vec)

        y_vec_front = np.zeros(len(word_to_index))
        y_vec_back = np.zeros(len(word_to_index))
        for j in range(1, window_size):
            y_vec_front[word_to_index[window_front[j]]] += 1
            y_vec_back[word_to_index[window_back[j]]] += 1

        trainY.append(y_vec_back)
        trainY.append(y_vec_front)
    
    return (np.array(trainX), np.array(trainY))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python P3C.py <path_to_textfile> <num_latent_factors>")
        sys.exit(1)

    filename = sys.argv[1]
    num_latent_factors = int(sys.argv[2])
    # Load in a list of words from the specified file; remove non-alphanumeric characters
    # and make all chars lowercase.
    sample_text = load_word_list(filename)

    # Create dictionary mapping unique words to their one-hot-encoded index
    word_to_index = generate_onehot_dict(sample_text)
    # Create training data using default window size
    trainX, trainY = generate_traindata(sample_text, word_to_index)
    # TODO: 1) Create and train model in Keras.      
    # vocab_size = number of unique words in our text file. Will be useful when adding layers
    # to your neural network
    vocab_size = len(word_to_index)
    model = Sequential()

    hidden_layer = Dense(10)
    model.add(layer)
    output_layer = Dense(10)
    model.add(output_layer)
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer="adam",
                  metrics="accuracy")
    fit = modelD.fit(trainX, trainY, batch_size=64, epochs=10,
    verbose=0)
    # TODO: 2) Extract weights for hidden layer, set <weights> variable below
    # weights = None
    input_weights = hidden_layer.get_weights()
    output_weights = output_layer.get_weights()

    # Find and print most similar pairs
    similar_pairs = most_similar_pairs(weights, word_to_index)
    for pair in similar_pairs[:30]:
        print(pair)
