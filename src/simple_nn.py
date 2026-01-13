import numpy as np


class SimpleNeuralNetwork:
    """
    Single-layer neural network (no hidden layer)
    trained with gradient descent and sigmoid activation.
    """

    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

        # parameters (initialized in fit)
        self.W = None  # (D, K)
        self.b = None  # (K,)

    # -------------------------------------------------
    # Forward pass components
    # -------------------------------------------------
    def sigmoid(self, a):
        """
        :param self: object of SimpleNeuralNetwork
        :param a: pre-activation matrix 
        :return y_hat: sigmoid output matrix
        """
        y_hat = 1/(1+np.exp(-a))
        return y_hat

    def forward(self, X):
        """
        :param X: Input matrix (N,D)    (N=samples, D=input features)
        :return a: pre-activation matrix  (N,K) (K=outputs/classes)
        :return y_hat: sigmoid output matrix (N,K)
        """
        a = X @ self.W + self.b
        y_hat = self.sigmoid(a)
        return a, y_hat

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    def compute_loss(self, y_hat, y):
        """
        :param y_hat: sigmoid output matrix (N,K)
        :param y: ground truth (N,K)
        :return loss: MSE (float)
        """
        loss = np.mean(np.square(y_hat-y))
        return loss

    # -------------------------------------------------
    # Backward pass (gradient computation)
    # -------------------------------------------------
    def backward(self, X, y, a, y_hat):
        """
        Computes gradients dW and db for one gradient step.

        X: (N, D)
        y: (N, K)
        a: (N, K)   (not strictly needed here, but kept for later extensions)
        y_hat: (N, K)

        returns:
          dW: (D, K)
          db: (K,)
        """
   
        # Compute the error term delta = ∂E / ∂a
        # ∂E / ∂y_hat = (y_hat-y)
        # ∂y_hat / ∂a = σ(a)(1-σ(a))
        loss_error = (y_hat-y)
        sigmoid_slope = y_hat * (1-y_hat)
        delta = loss_error * sigmoid_slope

        # Compute the gradient w.r.t. the weights
        dW = np.transpose(X) @ delta / X.shape[0]

        # Compute the gradient w.r.t. the bias
        db = np.sum(delta,axis=0) / delta.shape[0]

        return dW, db

    # -------------------------------------------------
    # Gradient descent step (EXPLICIT)
    # -------------------------------------------------
    def gradient_step(self, dW, db):
        """
        Performs one gradient descent update
        """
        self.W -= self.lr * dW
        self.b -= self.lr * db

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def fit(self, X, y):
        """
        Train the model using full-batch gradient descent.

        X: (N, D) input data
        y: (N, K) one-hot labels
        """
        # Init the weight matrices
        self.W = 0.01 * np.random.randn(X.shape[1], y.shape[1])
        self.b = np.zeros(y.shape[1])

        self.loss_history = []

        # Realize gradient descent (iteratively)
        for _ in range(self.epochs):
            # TODOs: 
            #   Forward pass
            a,y_hat = self.forward(X)
            #   Compute loss
            loss = self.compute_loss(y_hat,y)
            self.loss_history.append(loss)
            #   Backward pass
            dW,db = self.backward(X,y,a,y_hat)
            #   Gradient step
            self.gradient_step(dW,db)
            

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, X):
        """
        Predict class labels (0..K-1).
        """
        _, y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)
    
    def predict_proba(self, X):
        """
        Predict probabilities per class (sigmoid outputs).
        """
        _, y_hat = self.forward(X)
        return y_hat