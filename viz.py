import matplotlib.pyplot as plt
import seaborn as sns

def plot_numeric_histograms(df, n_cols = 3):
    
    columns = df.select_dtypes(include='number').columns
    
    n_rows = (len(columns) // n_cols) + (len(columns) % n_cols > 0)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Plot histograms for each column using the specified hist method
    for i, col in enumerate(columns):
        ax = axes[i]
        ax.hist(df[col].dropna(), edgecolor='black', color='orange')
        ax.set_title(f'Histogram of {col}', fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    # Remove empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def plot_histogram(df, column, rotation = 45):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], discrete=True, color='orange', kde=False)
    plt.title(f'Histogram of {column}', fontsize=14)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=rotation)
    plt.show()