import matplotlib.pyplot as plt
import seaborn as sns

def correlation_matrix(X, num, cmap='Blues'):
    X_num = X[num]
    
    # Compute the absolute correlation matrix
    corr_matrix = X_num.corr().abs()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()


from sklearn.feature_selection import SelectKBest, chi2

def chi_squared(X, y, categ, threshold=0.05):
    X_categ = X[categ]
    
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_categ), 
                            columns=X_categ.columns)
    
    # Fit the chi-squared selector
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(X_scaled, y)

    # Get Chi-squared scores
    chi2_scores = chi2_selector.scores_

    # Create a DataFrame for scores
    scores_df = pd.DataFrame({
        'Feature': X_categ.columns,
        'Chi2 Score': chi2_scores
    })

    # Filter features based on the threshold
    selected_features = scores_df[scores_df['Chi2 Score'] > threshold]['Feature']
    
    # Plot the Chi-squared scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Chi2 Score', y='Feature', data=scores_df)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Chi-squared Scores for Features')
    plt.xlabel('Chi-squared Score')
    plt.ylabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nInitial Features:\n")
    print(X_categ.columns.tolist())
    print("\nFinal Decision for Categorical Features:\n")
    print(selected_features.tolist())
