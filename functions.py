import matplotlib.pyplot as plt

def plot_histogram(data, xlabel, ylabel, title, color='orange'):
    """
    Plots a histogram of the given data.

    Parameters:
        data (array-like): The data to be plotted.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the histogram.
        color (str): The color of the histogram bars. Default is 'blue'.
    """
    plt.hist(data, edgecolor='black', color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()