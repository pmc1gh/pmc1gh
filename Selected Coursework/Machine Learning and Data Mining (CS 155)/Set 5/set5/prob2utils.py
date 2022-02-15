# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return \
        eta * (reg * Ui - Vj * (Yij - np.dot(np.transpose(Ui), Vj)))

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return \
        eta * (reg * Vj - Ui * (Yij - np.dot(np.transpose(Ui), Vj)))

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    squared_sum = 0
    for k in range(len(Y)):
        i = Y[k][0] - 1
        j = Y[k][1] - 1
        Yij = Y[k][2]
        squared_sum += np.square(Yij - np.dot(U[i], V[j]))
    
    err_total = 0.5 * reg * (np.square(np.linalg.norm(U))
                     + np.square(np.linalg.norm(V))) \
                + 0.5 * squared_sum
    return err_total / len(Y)


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """

    U = np.random.rand(M, K) - 0.5
    V = np.random.rand(N, K) - 0.5
    # print(np.shape(U))

    mse = 100
    for epoch in range(max_epochs):
        print(mse)
        if mse < eps:
            break
        # Iterate through the points in a random order and update U and V accordingly.
        indices = np.random.permutation(len(Y))
        for k in indices:
            i = Y[k][0] - 1
            j = Y[k][1] - 1
            Yij = Y[k][2]
            U[i] = U[i] - grad_U(U[i], Yij, V[j], reg, eta)
            V[j] = V[j] - grad_V(V[j], Yij, U[i], reg, eta)
        mse = get_err(U, V, Y, reg)
    
    return U, V, mse
