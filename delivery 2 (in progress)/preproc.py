from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fill_dates(train_df, other_dfs, feature_prefix):

    # Define column names
    year_col = f'{feature_prefix} Year'
    month_col = f'{feature_prefix} Month'
    day_col = f'{feature_prefix} Day'
    
    # Calculate medians from the training dataframe
    accident_med = {
        year_col: round(train_df[year_col].median()),
        month_col: round(train_df[month_col].median()),
        day_col: round(train_df[day_col].median())
    }
    
    # Fill missing values and convert to integer type in the training set
    for col, med in accident_med.items():
        train_df[col].fillna(med, inplace=True)
        train_df[col] = train_df[col].astype('Int64')
    
    # Apply the same transformations to the other datasets
    for df in other_dfs:
        for col, med in accident_med.items():
            df[col].fillna(med, inplace=True)
            df[col] = df[col].astype('Int64')


def fill_dow(dataframes, feature_prefix):
    # Define column names
    year_col = f'{feature_prefix} Year'
    month_col = f'{feature_prefix} Month'
    day_col = f'{feature_prefix} Day'
    dayofweek_col = f'{feature_prefix} Day of Week'
    
    # Loop through the provided dataframes to process each one
    for df in dataframes:
        # Identify rows where the 'Day of Week' column is missing
        missing_dayofweek = df[dayofweek_col].isnull()
        
        # If there are missing values in 'Day of Week'
        if missing_dayofweek.any():
            # Create a temporary 'Accident Date' column by combining Year, Month, and Day
            df.loc[missing_dayofweek, 'TEMP Accident Date'] = pd.to_datetime(
                df.loc[missing_dayofweek, [year_col, month_col, day_col]]
                .astype(str)                   
                .agg('-'.join, axis=1),        
                errors='coerce')
            
            # Fill the missing 'Day of Week' using the newly created 'TEMP Accident Date'
            df.loc[missing_dayofweek, dayofweek_col] = df.loc[missing_dayofweek, 'TEMP Accident Date'].dt.dayofweek
            
            # Drop the temporary column after it's no longer needed
            df.drop(columns=['TEMP Accident Date'], inplace=True, errors='ignore')
        
        # Ensure the 'Day of Week' column has the correct integer type
        df[dayofweek_col] = df[dayofweek_col].astype('Int64')


def fill_birth_year(dfs):
    # Define fixed column names
    year_col = 'Accident Date Year'
    age_col = 'Age at Injury'
    birth_year_col = 'Birth Year'

    # Process the other DataFrames
    for df in dfs:
        mask = df[year_col].notna() & df[age_col].notna() & \
               (df[birth_year_col].isna() | (df[birth_year_col] == 0))
        df.loc[mask, birth_year_col] = df[year_col] - df[age_col]


def ball_tree_impute(dfs, target, n_neighbors=5):

    for df in dfs:
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
        df[target] = df[target].combine_first(imputed_values)



## Outliers


def detect_outliers_iqr(df, missing_threshold):
    missing_col = []
    outliers_indices = set()
    bounds = {}  
    
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Store the bounds
        bounds[column] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
        
        # Identify outliers
        outlier_data = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_indices.update(outlier_data.index)
        
        missing = len(outlier_data) / len(df) * 100
        
        # Print the number of outliers
        print(f'Column: {column} - Number of Outliers: {len(outlier_data)}')
        print(f'Column: {column} - % of Outliers: {missing:.2f}% \n')
        
        if missing > missing_threshold:
            missing_col.append(column)
        
        # Boxplot for each column
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=column, color='orange', showfliers=False)  
        sns.stripplot(
            data=outlier_data, 
            x=column, 
            color='red', 
            jitter=True, 
            label='Outliers'
        )
        plt.title(f'Boxplot with Outliers for {column}')
        plt.legend()
        plt.show()
    
    print(f'Columns with more than {missing_threshold}% Outliers:')        
    print(missing_col)
    
    return bounds  
