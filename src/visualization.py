import matplotlib.pyplot as plt
import numpy as np


# Exercise 1 - plotting the training curve
def plot_training_curve(loss_history, path):
    """
    Plot training loss over epochs and save to file.

    Parameters
    ----------
    loss_history : list or np.ndarray
        Loss values collected during training.
    path : str
        Directory and file name for the output PDF file.

    Returns
    -------
    str
        Path to the saved figure.
    """

    loss_history = np.array(loss_history)
    epochs = np.arange(len(loss_history))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs,loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over epochs")
    plt.grid(True)

    plt.savefig(path)
    plt.close()


    return path

# Exercise 1 - Visualization of the data set and decision regions
def plot_decision_regions(model, X, y, path):
    """
    Plot decision regions and training data points

    Parameters
    ----------
    model : 
        fitted model, can be used to predict any point in the plane
    X:
        training inputs
    y:
        training targets
    path : str
        Directory and file name for the output PDF file.

    Returns
    -------
    str
        Path to the saved figure.
    """
    x_min = np.min(X[:,0]) - 0.5
    x_max = np.max(X[:,0]) + 0.5
    y_min = np.min(X[:,1]) - 0.5
    y_max = np.max(X[:,1]) + 0.5

    x_grid = np.linspace(x_min,x_max,400)
    y_grid = np.linspace(y_min,y_max,400)
    xx,yy = np.meshgrid(x_grid,y_grid)

    gridpoints = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(gridpoints)
    plot_input = np.reshape(predictions, xx.shape)

    # If y is one-hot encoded, convert to class labels
    if y.ndim == 2:
        y = np.argmax(y, axis=1)



    plt.figure(figsize=(6, 6))
    plt.contourf(xx,yy,plot_input)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20)
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title("Linear decision regions")

    plt.savefig(path)
    plt.close()

    return path

# Exercise 3 - Visualize accuracies depending on the hidden layer size
# at the end of training
def plot_hidden_size_vs_accuracy(
    hidden_sizes,
    train_accuracies,
    test_accuracies,
    path
):
    """
    Plot training and test accuracy as a function of hidden layer size.

    Parameters
    ----------
    hidden_sizes : list[int]
        Number of neurons in the hidden layer.
    train_accuracies : list[float]
        Training accuracies for each hidden size.
    test_accuracies : list[float]
        Test accuracies for each hidden size.
    path : str
        Directory and file name of the output PDF file.

    Returns
    -------
    str
        Path to the saved PDF file.
    """
    # TODO: create a plot that visualizes the training
    # and test accuracies (given on the y-axis)
    # with respect to the hidden layer size.

    return path

def plot_mean_learning_curve(losses, accuracies, label, path):
    """
    Plot mean ± std of loss and accuracy over epochs.

    losses:       list or array, shape (n_runs, epochs)
    accuracies:   list or array, shape (n_runs, epochs)
    label:        str (e.g. 'SGD, batch_size=8')
    path:         output PDF path
    """
    losses = np.array(losses)
    accuracies = np.array(accuracies)

    loss_mean = losses.mean(axis=0)
    loss_std = losses.std(axis=0)

    acc_mean = accuracies.mean(axis=0)
    acc_std = accuracies.std(axis=0)

    epochs = np.arange(len(loss_mean))

    fig, ax1 = plt.subplots()

    # ---- loss (left axis) ----
    ax1.plot(epochs, loss_mean, label=f"{label} – loss", color="tab:blue")
    ax1.fill_between(
        epochs,
        loss_mean - loss_std,
        loss_mean + loss_std,
        alpha=0.3,
        color="tab:blue"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # ---- accuracy (right axis) ----
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc_mean, label=f"{label} – accuracy", color="tab:orange")
    ax2.fill_between(
        epochs,
        acc_mean - acc_std,
        acc_mean + acc_std,
        alpha=0.3,
        color="tab:orange"
    )
    ax2.set_ylabel("Accuracy")

    # ---- legend ----
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# Exercise 4: Generate a plot comparing accuracies for different batch sizes.
def plot_accuracy_comparison(acc_list_1, acc_list_2,
                             label_1, label_2,
                             path):
    """
    Compare two accuracy learning curves (mean and std).

    acc_list_1 : list of runs, each run is a list of accuracies over epochs
    acc_list_2 : second list of runs, each run is a list of accuracies over epochs
    """
    # TODO: create a plot that shows
    # for the two different conditions 
    # how the accuracies evolve over time
    # 1) For each condition you get a list of runs,
    #    Each run is a list of accuracies over time.
    #    In a first step compute mean and std.dev. values
    # 2) Generate for each of the two acc_lists / conditions
    #    a line as a plot (and an area visualizing the std.dev.)

    return path
