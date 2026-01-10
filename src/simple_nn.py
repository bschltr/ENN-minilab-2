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
        # TODO Implement sigmoid activation
        return np.zeros_like(a)

    def forward(self, X):
        # TODO Realize the forward pass
        # Currently: Returns arrays with correct shapes, but no real computation.
        N = X.shape[0]
        output_dim = 3
        # Return both: Activation values and after application of activation function (output)
        a = np.zeros((N, output_dim))
        y_hat = np.zeros((N, output_dim))
        return a, y_hat

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    def compute_loss(self, y_hat, y):
        # TODO Compute the loss (MSE)
        # Currently simply returning a value.
        return float(np.mean(y_hat))

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
        # TODO: Realize the backward pass - given in steps.
        # TODO 1:
        # Compute the error term delta = ∂E / ∂a
        # Hint:
        #   - start from (y_hat - y)
        #   - multiply with the derivative of the sigmoid
        #   - sigmoid'(a) can be expressed using y_hat
        delta = np.zeros_like(y_hat)

        # TODO 2:
        # Compute the gradient w.r.t. the weights
        # Hint:
        #   - you will have to use the Input values and delta
        #   - if you want to deal with batches: average over the batch size
        dW = np.zeros_like(self.W)

        # TODO 3:
        # Compute the gradient w.r.t. the bias
        db = np.zeros_like(self.b)

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
            #   Compute loss
            #     and self.loss_history.append(loss)
            #   Backward pass
            #   Gradient step
            pass

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