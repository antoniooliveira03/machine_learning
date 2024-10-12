import matplotlib.pyplot as plt
import pandas as pd

def plot_histogram(data, xlabel, ylabel, title, color='lightblue'):
    """
    Plots a histogram of the given data.

    Parameters:
        data (array-like): The data to be plotted.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the histogram.
        color (str): The color of the histogram bars. Default is 'lightblue'.
    """
    plt.hist(data, edgecolor='black', color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def boxplots(data, color='lightblue'):
    """
    Generate boxplotsfor numeric columns in the provided data.

    Args:
      data (pandas.DataFrame): The input data containing numeric columns.
      graph (str): The type of graph to generate. Options: 'boxplot' or 'histogram'.
      color (str): The color of the plot. Default is 'lightblue'.
    """

    for column in data.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            plt.boxplot(data[column], vert=False, patch_artist=True, boxprops=dict(facecolor=color))
            plt.title(f'Boxplot of {column}')
            plt.yticks([])
            plt.show()
            
        else:
            continue