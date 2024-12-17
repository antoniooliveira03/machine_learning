# Machine Learning Course for DS Master 2024/2025 Repository

This Repository is divided in 2 main folders.
1. Delivery 1 - Notebooks and python files used for the first delivery
2. Delivery 2 - Notebooks and python files used for the second delivery

Other folders were used in the development of the project, but are not relevant for the final solution.

Focusing on the Second and Final delivery we have the following Notebooks:
1. Exploratory Data Analysis
- In this notebook we explore our data in-depth by looking at statistics and visual representations of each feature. We also created new features that could later be useful.
2. Holdout Method
- This notebook consists of the implementation of the Holdout Method, after which Encoding is applied, Missing values are addressed and Outliers are dealt with. We also performed Feature Selection and implemented ML models. With this notebook we aimed to understand which were the more relevant features and to evaluate which models performed best, as well as their best hyperparameters, in order to later use these conclusions in the next notebook.
3. Cross Validation
- This notebook consists of the implementation of a Cross Validation Method - Stratified K-Fold. Here the same preprocessing steps are applied as in the previous notebook, but only the previously selected models and features are used. In this notebook we also implemented Over and Undersampling techniques. 

Focusing on the Second and Final delivery we have the following Python files:
- map_ - produces a map in notebook 01 by using a JSON file
- models - include all implemented models as well as the Holdout and Cross Validation functions
- tuning - includes hyperparameter tuning functions
- metrics - includes 2 different metrics functions
- utils - includes functions used in Notebook 01
- utils2 - includes all preprocessing functions
- viz - includes all visualisations functions
- play_song - includes a function to play a song (useful when running models for a long time)
