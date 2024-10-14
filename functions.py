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



def plot_missing_values_bar(data, xlabel, ylabel, title):
    """
    Plots a bar chart of missing values for each column.

    Parameters:
        data (Series): A Pandas Series containing column names and their corresponding missing values.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the bar chart.
    """
    plt.bar(data.index, data.values, edgecolor='white', color='lightblue')
    plt.xticks(rotation=45, ha='right')  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()  
    plt.show()