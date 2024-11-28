import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_histogram(data, xlabel, ylabel, title, color='orange'):
    """
    Plots a histogram of the given data.

    Parameters:
        data (array-like): The data to be plotted.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the histogram.
        color (str): The color of the histogram bars. Default is 'orange'.
    """
    plt.hist(data, edgecolor='black', color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_boxplot(data, x, y, xlabel, ylabel, title, xticks_labels=None):
    """
    Creates a boxplot to compare the distribution of a variable across categories.

    Parameters:
        data (DataFrame): The data containing the variables to plot.
        x (str): The column name for the categorical variable (x-axis).
        y (str): The column name for the numerical variable (y-axis).
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
        xticks_labels (list): Custom labels for the x-axis ticks (default is None).
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y, color='orange')  # Set to orange color
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Apply custom x-tick labels if provided
    if xticks_labels:
        plt.xticks(ticks=range(len(xticks_labels)), labels=xticks_labels)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def boxplots(data, color='orange'):
    """
    Generate boxplotsfor numeric columns in the provided data.

    Args:
      data (pandas.DataFrame): The input data containing numeric columns.
      graph (str): The type of graph to generate. Options: 'boxplot' or 'histogram'.
      color (str): The color of the plot. Default is 'orange'.
    """

    for column in data.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            plt.boxplot(data[column], vert=False, patch_artist=True, boxprops=dict(facecolor='orange', color='black'),
                    medianprops=dict(color='black'))
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
                    patch_artist=True, boxprops=dict(facecolor='orange', color='black'),
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
    plt.bar(data.index, data.values, edgecolor='white', color='orange')
    plt.xticks(rotation=45, ha='right')  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()  
    plt.show()


def plot_line(x, y, xlabel, ylabel, title):
    """
    Plots a line graph with the provided data.

    Parameters:
        x (array-like): The data for the x-axis.
        y (array-like): The data for the y-axis.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the line plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='orange', marker='o')  # Orange color with markers
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_subplots(data, columns, titles, xlabels, ylabels):
    """
    Creates a row of count plots (subplots) for the specified columns.

    Parameters:
        data (DataFrame): The data containing the columns to plot.
        columns (list): A list of column names to plot.
        titles (list): A list of titles for each subplot.
        xlabels (list): A list of x-axis labels for each subplot.
        ylabels (list): A list of y-axis labels for each subplot.
    """
    num_plots = len(columns)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    for i, column in enumerate(columns):
        sns.countplot(data=data, x=column, ax=axes[i], color='orange')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel(ylabels[i])

    plt.tight_layout()
    plt.show()


def plot_regression(data, x, y, xlabel, ylabel, title, scatter_alpha=0.5):
    """
    Creates a regression plot to visualize the relationship between two variables.

    Parameters:
        data (DataFrame): The data containing the variables to plot.
        x (str): The column name for the x-axis variable.
        y (str): The column name for the y-axis variable.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
        scatter_alpha (float): The transparency level for scatter points (default is 0.5).
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x, y=y, data=data, scatter_kws={'alpha': scatter_alpha}, color='orange')  # Set to orange
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_boxenplot(data, x, y, xlabel, ylabel, title):
    """
    Creates a boxen plot to visualize the distribution of a numerical variable across categories.

    Parameters:
        data (DataFrame): The data containing the variables to plot.
        x (str): The column name for the categorical variable (x-axis).
        y (str): The column name for the numerical variable (y-axis).
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(12, 6))
    sns.boxenplot(x=x, y=y, data=data, color='orange')  # Set to orange color
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_bar(data, column, xlabel, ylabel, title, color='orange'):
    """
    Creates a bar chart to display the proportion of each category in a specified column.

    Parameters:
        data (DataFrame): The data containing the column to plot.
        column (str): The column name for which to plot the proportions.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
        color (str): The color of the bars (default is 'orange').
    """
    data[column].value_counts(normalize=True).plot(kind='bar', color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
