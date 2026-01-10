import numpy as np

class MLPOneHiddenLayer:
    """
    MLP with exactly one hidden layer and sigmoid activations.
    - Output uses sigmoid per class (one-hot labels expected).
    - Loss: MSE (same style as the simple_nn baseline).
    """

    def __init__(self, hidden_dim=5, lr=0.01, epochs=1000):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

        self.input_dim = 2
        self.output_dim = 3

        # parameters (set in _init_params)
        # Note: This is realized as 
        # - input dimensions in first dimension
        # - hidden layer in second dimension
        # in an np.array.
        # Leads to computation order: Input array multiplied by matrix.
        self.W1 = None  # (D, H)
        self.b1 = None  # (H,)
        self.W2 = None  # (H, K)
        self.b2 = None  # (K,)

    # -------------------------
    # Init function
    # -------------------------
    def reset_weights(self):
        """
        Initialize weights and biases.
        """
        # Note: This is realized as 
        # - input dimensions in first dimension
        # - hidden layer in second dimension
        # in an np.array.
        # Leads to computation order: Input array X applied to matrix W.
        self.W1 = 0.01 * np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros(self.hidden_dim)

        self.W2 = 0.01 * np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros(self.output_dim)
        

    # -------------------------------------------------
    # Forward pass components
    # -------------------------------------------------
    def sigmoid(self, a):
        # TODO Implement sigmoid activation
        return np.zeros_like(a)

    def forward(self, X):
        """
        Forward pass.

        Returns a dict containing all intermediate activations
        needed for backpropagation.
        """
        # TODO 1: compute a1 activations in hidden layer neurons
        a1 = np.zeros((X.shape[0], self.hidden_dim))

        # TODO 2: apply activation function -> z1
        z1 = np.zeros_like(a1)

        # TODO 3: compute a2 activations in output neurons
        a2 = np.zeros((X.shape[0], self.output_dim))

        # TODO 4: compute y_hat output (apply activation function)
        y_hat = np.zeros_like(a2)

        activations = {
            "X": X,
            "a1": a1,
            "z1": z1,
            "a2": a2,
            "y_hat": y_hat
        }
        return activations, y_hat


    # -------------------------------------------------
    # Loss
    # -------------------------------------------------
    def compute_loss(self, y_hat, y):
        """
        MSE loss.
        """
        # TODO Compute the loss
        return float(np.mean(y_hat))

    # -------------------------------------------------
    # Backward pass (gradient computation)
    # -------------------------------------------------
    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid σ'(a) expressed via σ(a) = z.
        """
        # TODO Implement derivative of sigmoid using z
        return np.zeros_like(z)
    
    def backward(self, y, activations):
        """
        Backpropagation using activations from forward pass.

        Returns:
          grads (dict) with keys: W1, b1, W2, b2
        """
        X = activations["X"]
        z1 = activations["z1"]
        y_hat = activations["y_hat"]

        N = X.shape[0]

        # -----------------------------------------
        # TODO 1: Compute output layer error term (delta2)
        # Hint: sigmoid'(a2) can be expressed via y_hat
        # -----------------------------------------
        delta2 = np.zeros_like(y_hat)

        # -----------------------------------------
        # TODO 2: Gradients for W2 and b2
        # -----------------------------------------
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        # -----------------------------------------
        # TODO 3: Hidden layer error term (delta1)
        # delta1 = (delta2 @ W2^T) * sigmoid'(a1)
        # Hint: sigmoid'(a1) can be expressed via z1
        # -----------------------------------------
        delta1 = np.zeros_like(z1)

        # -----------------------------------------
        # TODO 4: Gradients for W1 and b1
        # -----------------------------------------
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return grads

    # -------------------------------------------------
    # Gradient descent step (EXPLICIT)
    # -------------------------------------------------
    def gradient_step(self, grads):
        """
        Performs one gradient descent update
        """
        # TODO Apply parameter updates using self.lr
        # self.W1 -= ...
        # self.b1 -= ...
        # self.W2 -= ...
        # self.b2 -= ...
        pass

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def fit(self, X, y, batch_size=8):
        """
        Train the model using gradient descent.

        X: (N, D)
        y: (N, K)
        batch_size:
            use a batch size - otherwise you will require much more epochs 
            to see a training effect.
        """
        self.reset_weights()
        self.loss_history = []

        N = X.shape[0]

        for epoch in range(self.epochs):
            # shuffle once per epoch
            perm = np.random.permutation(N)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            epoch_loss = 0.0

            # mini-batch loop
            for start in range(0, N, batch_size):
                end = start + batch_size
                Xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                # TODOs: 
                #   forward pass
                #   compute the losses: for the epoch and the batch
                #   backward pass
                #   update / gradient step

            self.loss_history.append(epoch_loss / N)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, X):
        _, y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)
    
    # --------------------------------------------------
    # predefined weights (for checking forward pass)
    # --------------------------------------------------
    def set_predefined_weights(self):
        W1, b1, W2, b2 = self.predefined_spiral_weights()

        self.W1 = W1.copy()
        self.b1 = b1.copy()
        self.W2 = W2.copy()
        self.b2 = b2.copy()

    @staticmethod
    def predefined_spiral_weights():
        b1 = np.array([13.955, -1.079, -1.420, -9.452, -6.745])
        # Note: This is realized as 
        # - input dimensions in first dimension
        # - hidden layer in second dimension
        # in an np.array.
        # Leads to computation order: Input array multiplied by matrix.
        W1 = np.array([
            [ 5.290,  24.748,  9.823, -22.713, 13.622],
            [21.145, -28.266,  0.001,   1.745,  8.618],
        ])

        b2 = np.array([14.265, -0.281, -25.521])
        W2 = np.array([
            [-17.921,   3.912,  23.039],
            [-10.631,  -5.133,  15.726],
            [ 27.089, -17.227, -18.574],
            [ -0.646, -12.115,   7.881],
            [-24.915,  32.302, -11.516],
        ])

        return W1, b1, W2, b2