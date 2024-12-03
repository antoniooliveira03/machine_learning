import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from wordcloud import WordCloud



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



def plot_pairwise_relationship(df):
    # Select only numerical columns
    num_columns = df.select_dtypes(include=['number']).columns

    # Create a pairplot with only the lower triangle
    g = sns.pairplot(df[num_columns], kind='scatter', 
                 hue=None, plot_kws={'s': 10, 'color': 'orange'}, 
                 corner=True) 
    
    # Update the diagonal plots to be orange
    for ax in g.diag_axes:
        for patch in ax.patches:
            patch.set_facecolor('orange')

        
    plt.show()


def plot_crosstab(df, column1, column2, annot_kws={"rotation": 45}):
    # Create the crosstab
    crosstab = pd.crosstab(df[column1], df[column2])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap='Oranges', annot_kws=annot_kws)
    plt.title(f'{column1} vs {column2}')
    plt.show()


def plot_categ_cont(df, categorical, continuous, n_cols=3):
    # Calculate the number of rows needed based on number of categories and columns
    n_cats = len(categorical)
    n_cont = len(continuous)
    
    # Calculate number of rows required
    n_rows = int(np.ceil(n_cats / n_cols))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6, n_rows * 6))
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    plot_idx = 0
    for i, cat in enumerate(categorical):
        for j, cont in enumerate(continuous):
            if plot_idx < len(axes):
                sns.boxplot(x=cat, y=cont, data=df, palette='Oranges', ax=axes[plot_idx])
                axes[plot_idx].set_title(f'{cat} vs {cont}')
                axes[plot_idx].tick_params(axis='x', rotation=45)
                plot_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_cat_vs_num(df, categorical_column, numerical_column, plot_type="box"):
    plt.figure(figsize=(10, 6))
    palette = 'Oranges'
    
    category_order = sorted(df[categorical_column].unique())

    if plot_type == "box":
        sns.boxplot(data=df, x=categorical_column, y=numerical_column, 
                    palette=palette, order=category_order)
    elif plot_type == "bar":
        sns.barplot(data=df, x=categorical_column, y=numerical_column, estimator="mean", 
                    palette=palette, order=category_order)
        
    plt.title(f'{categorical_column} vs {numerical_column}')
    plt.show()



def generate_wordcloud(df, column_name, title='Word Cloud', width=800, height=400, colormap='Oranges', background_color='white', max_words=None):
    # Step 1: Combine all text in the specified column into a single string
    all_text = " ".join(df[column_name].astype(str).tolist()).lower()
    
    # Step 2: Tokenize the text
    words = all_text.split()

    # Step 3: Count the word frequencies
    word_counts = Counter(words)

    # Step 4: Generate the word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words
    ).generate_from_frequencies(word_counts)

    # Step 5: Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()
