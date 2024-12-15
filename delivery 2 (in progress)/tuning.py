import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Initialize DataFrame 
search_results_df = pd.DataFrame()

def hyperparameter_search(model, param_grid, search_type, 
                          X_train, y_train, scoring='accuracy', 
                          cv=3, n_iter=10, random_state=42,
                          reset=False):
    
    # Use the global DataFrame to save results
    global search_results_df  
    
    if reset:
        search_results_df = pd.DataFrame()
    
    #  Random Search
    if search_type == "random":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            verbose=1,
            random_state=random_state
        )

    #  Grid Search
    elif search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            verbose=1
        )
        
    # Fit
    search.fit(X_train, y_train)
    
    # Best Parameters
    best_params_df = pd.DataFrame([search.best_params_])
    best_params_df["Search Type"] = "RandomizedSearchCV" if search_type == "random" else "GridSearchCV"
    best_params_df["Number of Fits"] = len(search.cv_results_["params"])
    best_params_df["Model"] = str(model).split("(")[0] 
    
    # Append to DataFrame
    search_results_df = pd.concat([search_results_df, best_params_df], ignore_index=True)
    
    return search_results_df.T
