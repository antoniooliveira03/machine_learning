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


def boxplot_out(data, columns, ncols=2):
    
    num_columns = len(columns)
    nrows = (num_columns + ncols - 1) // ncols  # Calculates the number of rows needed

    plt.figure(figsize=(12 * ncols, 8 * nrows))

    for i, column in enumerate(columns):
        # Calculate quartiles and IQR
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Determine the outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]

        # Create the box plot for the current column
        plt.subplot(nrows, ncols, i + 1)  # Create a subplot grid
        plt.boxplot(data[column], vert=False, widths=0.7,
                    patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
                    medianprops=dict(color='black'))

        # Scatter outliers
        plt.scatter(outliers, [1] * len(outliers), color='red', marker='o', label='Outliers')

        # Customize the plot
        plt.title(f'Box Plot of {column} with Outliers')
        plt.xlabel('Value')
        plt.yticks([])
        plt.legend()

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()



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