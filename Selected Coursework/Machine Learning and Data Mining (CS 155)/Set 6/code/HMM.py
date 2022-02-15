########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
from copy import deepcopy

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # k = 1 Initialization
        for a in range(self.L):
            probs[1][a] = self.A_start[a] * self.O[a][x[0]]
            seqs[1][a] = str(a)
        
        for j in range(2, len(probs)):
            for a in range(len(probs[0])):
                # Find the maximum current probability
                p_list = []
                for yi in range(len(probs[0])):
                    p_list.append((probs[j-1][yi]
                        *self.A[yi][a]*self.O[a][x[j-1]], yi))
                p_list.sort()
                p_list.reverse()
                probs[j][a] = p_list[0][0]
                seqs[j][a] = seqs[j-1][p_list[0][1]] + str(a)

        max_seq = ''
        max_prob = -1
        for a in range(len(probs[0])):
            if max_prob < probs[len(probs)-1][a]:
                max_prob = probs[len(probs)-1][a]
                max_seq = seqs[len(seqs)-1][a]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialization
        for a in range(self.L):
            alphas[1][a] = self.A_start[a] * self.O[a][x[0]]
        
        for j in range(2, len(alphas)):
            Ca = 0
            for a in range(len(alphas[0])):
                total = 0
                for ap in range(len(alphas[0])):
                    total += \
                        alphas[j-1][ap]*self.A[ap][a]
                total *= self.O[a][x[j-1]]
                alphas[j][a] = total
                Ca += total
            if normalize:
                for a in range(len(alphas[0])):
                    alphas[j][a] /= Ca

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialization
        for b in range(self.L):
            betas[M][b] = 1
        
        for j in range(len(betas)-2, 0, -1):
            Cb = 0
            for b in range(len(betas[0])):
                total = 0
                for bp in range(len(betas[0])):
                    total += \
                        betas[j+1][bp] * self.A[b][bp] * self.O[bp][x[j]]
                betas[j][b] = total
                Cb += total
            if normalize:
                for b in range(len(betas[0])):
                    betas[j][b] /= Cb

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        for a in range(self.L):
            for b in range(self.L):
                nA = 0
                for i in range(len(Y)):
                    for j in range(1, len(Y[i])):
                        nA += (Y[i][j] == b) and (Y[i][j-1] == a)
                dA = 0            
                for i in range(len(Y)):
                    for j in range(1, len(Y[i])):
                        dA += (Y[i][j-1] == a)
                self.A[a][b] = nA / dA

        # Calculate each element of O using the M-step formulas.

        for a in range(self.L):
            for w in range(self.D):
                nO = 0
                for i in range(len(Y)):
                    for j in range(len(Y[i])):
                        nO += (X[i][j] == w) and (Y[i][j] == a)
                dO = 0            
                for i in range(len(Y)):
                    for j in range(len(Y[i])):
                        dO += (Y[i][j] == a)
                self.O[a][w] = nO / dO


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        alphas_list = [[] for _ in range(len(X))]
        betas_list = [[] for _ in range(len(X))]
        
        oldA = deepcopy(self.A)

        total1_list = [[0] * (len(X[i]) + 1) for i in range(len(X))]
        total2_list = [[0] * (len(X[i]) + 1) for i in range(len(X))]
        total3_list = [[0] * (len(X[i]) + 1) for i in range(len(X))]

        for k in range(N_iters): # N_iters
            if k < 20 or k % 20 == 0:
                print(k)
            for j in range(len(X)):
                alphas_list[j] = self.forward(X[j], normalize=True)
                betas_list[j] = self.backward(X[j], normalize=True)

            oldA = deepcopy(self.A)
            
            for i in range(len(X)):
                for j in range(1, len(X[i]) + 1):
                    total2_list[i][j] = 0
                    total3_list[i][j] = 0
                    total1_list[i][j] = 0
                    for ap in range(self.L):
                        if j >= 1 and j < len(X[i]) + 1:
                            total3_list[i][j] += \
                                alphas_list[i][j][ap] \
                                * betas_list[i][j][ap]
                        if j >= 2 and j < len(X[i]) + 1:
                            total1_list[i][j] += \
                                alphas_list[i][j-1][ap] \
                                * betas_list[i][j-1][ap]
                        for bp in range(self.L):
                            if j >= 1 and j < len(X[i]):
                                total2_list[i][j] += \
                                    alphas_list[i][j][ap] \
                                    * self.O[bp][X[i][j]] \
                                    * oldA[ap][bp] \
                                    * betas_list[i][j+1][bp]
            
            for a in range(self.L):
                # Update A.
                for b in range(self.L):
                    nA = 0
                    dA = 0
                    for i in range(len(X)):
                        for j in range(1, len(X[i])):
                            nnA = alphas_list[i][j][a] \
                                  * self.O[b][X[i][j]] * oldA[a][b] \
                                  * betas_list[i][j+1][b]
                            # print(len(X[i]), len(total2_list[i]))
                            nA += nnA / total2_list[i][j]
                        for j in range(2, len(X[i]) + 1):
                            dA += alphas_list[i][j-1][a] \
                                  * betas_list[i][j-1][a] \
                                  / total1_list[i][j]
                    
                    self.A[a][b] = nA / dA
                

            # Update O.
            for a in range(self.L):
                for w in range(self.D):
                    nO = 0
                    dO = 0
                    for i in range(len(X)):
                        for j in range(1, len(X[i]) + 1):
                            nO += (X[i][j-1] == w) * alphas_list[i][j][a] \
                                    * betas_list[i][j][a] \
                                    / total3_list[i][j]
                            dO += alphas_list[i][j][a] \
                                    * betas_list[i][j][a] \
                                    / total3_list[i][j]
                    self.O[a][w] = nO / dO


    def generate_emission(self, M): # See lecture 14 slide 22
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        for i in range(M):
            if i == 0:
                y_i = random.randrange(0, self.L)
            else:
                # Initialize a random y and see where
                # it is located in the probability
                # distribution.
                y_dist = random.random()
                for j in range(self.L):
                    if y_dist <= self.A[y_i][j]:
                        y_i = j
                        break
                    y_dist -= self.A[y_i][j]
            
            # Initialize a random x and see where
            # it is located in the probability
            # distribution.
            x_dist = random.random()
            for j in range(self.D):
                if x_dist <= self.O[y_i][j]:
                    x_i = j
                    break
                x_dist -= self.O[y_i][j]
            
            emission.append(x_i)
            states.append(y_i)

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    random.seed(2019)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(2019)
    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
