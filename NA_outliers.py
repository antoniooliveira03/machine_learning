from sklearn.neighbors import NearestNeighbors
import numpy as np


def ball_tree_impute(df, target, n_neighbors=5):
    # Get all features except the target column
    features = df.columns.drop(target)
    
    # Separate rows with and without missing target values
    missing_mask = df[target].isna()
    non_missing_data = df[~missing_mask]
    missing_data = df[missing_mask]

    # Build a ball tree using all features except the target column
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    knn.fit(non_missing_data[features])

    # Find nearest neighbors for rows with missing values
    _, indices = knn.kneighbors(missing_data[features])

     # Initialize a Series to store imputed values
    imputed_values = pd.Series(index=df.index)

    # Impute missing values by averaging the target values of nearest neighbors
    for i, neighbor_indices in enumerate(indices):
        # Calculate the mean of the neighbors' target values
        mean_value = non_missing_data.iloc[neighbor_indices][target].mean()
        imputed_values[missing_data.index[i]] = mean_value

    # Combine the imputed values with the original target values
    result = df[target].combine_first(imputed_values)

    return result



def log_transform(X):
    return np.where(X > 0, np.log1p(X), X)




def custom_impute(df):
      
    for var_name in df.columns:
        
        if any(word in var_name for word in ['Year', 'Month', 'Day']) and var_name != 'Birth Year':
                df[var_name] = df[var_name].fillna(df[var_name].median())
        
        if var_name == 'Birth Year':
            # Only perform imputation for rows where both columns are not NaN and Birth Year is NaN or 0
            mask = df['Accident Year'].notna() & df['Age at Injury'].notna()
            df.loc[mask & (df['Birth Year'].isna() | (df['Birth Year'] == 0)), 
                    'Birth Year'] = df['Accident Year'] - df['Age at Injury']

            remaining_nans = df['Birth Year'].isna().sum()
            if remaining_nans > 0:
                median = df[var_name].median()
                df[var_name] = df[var_name].fillna(median)

        
        # Zip Code
        if var_name == 'Zip Code':
            df['Zip Code'] = df['Zip Code'].fillna(99999)

        # for all 'code' variables  
        if 'Code' in var_name and var_name != 'Zip Code':
            code_columns = df.filter(regex='Code$', axis=1).columns
            df[code_columns] = df[code_columns].fillna(0)

    
    df['Average Weekly Wage'] = ball_tree_impute(df, 'Average Weekly Wage')
          
    return df