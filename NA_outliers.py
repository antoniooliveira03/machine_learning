from sklearn.neighbors import NearestNeighbors
import numpy as np


def ball_tree_impute(df, target, n_neighbors=5):
    # Get all features except the target column
    features = df.columns.drop([target, 'Claim Injury Type'])
    
    # Separate rows with and without missing target values
    missing_mask = df[target].isna()
    non_missing_data = df[~missing_mask]
    missing_data = df[missing_mask]

    # Build a ball tree using all features except the target column
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    knn.fit(non_missing_data[features])

    # Find nearest neighbors for rows with missing values
    _, indices = knn.kneighbors(missing_data[features])

    # Impute missing values by averaging the target values of nearest neighbors
    imputed_values = [
        non_missing_data.iloc[neighbor_indices][target].mean() for neighbor_indices in indices
    ]
    
    # Assign the imputed values to the missing target values in the original DataFrame
    df.loc[missing_mask, target] = imputed_values

    return df


def custom_impute(df, var_name):
      
    
    if any(word in var_name for word in ['Year', 'Month', 'Day']) and var_name != 'Birth Year':
        df[var_name] = df[var_name].fillna(df[var_name].median())
         
    
    # Birth Year
    if var_name == 'Birth Year':
    # Only perform imputation for rows where both columns are not NaN and Birth Year is NaN or 0
        mask = df['Accident Year'].notna() & df['Age at Injury'].notna()
        df.loc[mask & (df[var_name].isna() | (df[var_name] == 0)), 
                   var_name] = df['Accident Year'] - df['Age at Injury']
    
  
    
    # Zip Code
    if var_name == 'Zip Code':
        df[var_name] = df[var_name].fillna(99999)
        
    # Wage
    if var_name == 'Average Weekly Wage':
        df = ball_tree_impute(df, var_name)
         
        
    # for all 'code' variables    
    code_columns = df.filter(regex='Code$', axis=1).columns
    df[code_columns] = df[code_columns].fillna(0)
        
    return df

def log_transform(X):
    return np.where(X > 0, np.log1p(X), X)
