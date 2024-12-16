# 
import time
import pandas as pd
import numpy as np

# Preprocessing
import utils2 as p

# Scalers
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler)

# Models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def run_model(model_name, X, y, params = None):

    if params is None:
        params = {}

    if model_name == 'LR':
        model = LogisticRegression(**params).fit(X, y)
    elif model_name == 'SGD':
        model = SGDClassifier(**params).fit(X, y)
    elif model_name == 'DT':
        model = DecisionTreeClassifier(**params).fit(X, y)
    elif model_name == 'RF':
        model = RandomForestClassifier(**params).fit(X, y)
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier(**params).fit(X, y)
    elif model_name == 'GBoost':
        model = GradientBoostingClassifier(**params).fit(X, y)
    elif model_name == 'XGB':
        model = XGBClassifier(**params).fit(X, y)
    elif model_name == 'MLP':
        model = MLPClassifier(**params).fit(X, y)
    elif model_name == 'GNB':  
        model = GaussianNB().fit(X, y)
    elif model_name == 'CNB':  
        model = CategoricalNB().fit(X, y)
    elif model_name == 'KNN':  
        model = KNeighborsClassifier(**params).fit(X, y)
    elif model_name == 'LGBM':  
        model = LGBMClassifier(**params).fit(X, y)
    elif model_name == 'SVM':  
        model = SVC(**params).fit(X, y)
    
    return model


def modeling(model_names, params,
             X_train, y_train, 
             X_val, y_val):
    
    results = {}
    
    for model_name in model_names:
        print(f"Training model: {model_name}")
        
        # Training
        model = run_model(model_name, X_train, y_train, params.get(model_name, {}))
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_precision = precision_score(y_train, y_train_pred, average='macro')
        train_recall = recall_score(y_train, y_train_pred, average='macro')
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        
        val_precision = precision_score(y_val, y_val_pred, average='macro')
        val_recall = recall_score(y_val, y_val_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        # Save results
        results[model_name] = {
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_macro_f1': train_f1,
            'val_macro_f1': val_f1
        }
        
        print(f"{model_name} - Train: Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Macro F1: {train_f1:.4f}")
        print(f"{model_name} - Validation: Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Macro F1: {val_f1:.4f}\n")
    
    return results


# KFOLD

def k_fold(method, X, y, test1, model_name, 
           params, enc, outliers = False,
           file_name = None):

    # Save metrics
    f1macro_train = []
    f1macro_val = []
    precision_train = []
    precision_val = []
    recall_train = []
    recall_val = []
    timer = []
    
    # Mapping
    label_mapping = {
        0: "1. CANCELLED",
        1: "2. NON-COMP",
        2: "3. MED ONLY",
        3: "4. TEMPORARY",
        4: "5. PPD SCH LOSS",
        5: "6. PPD NSL",
        6: "7. PTD",
        7: "8. DEATH"}
    
    test_preds = np.zeros((len(test1), len(label_mapping)))


    # For each fold
    for train_index, val_index in method.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        test = test1.copy()

        start_time = time.time()
        
        # ENCODING
        X_train['Alternative Dispute Resolution Bin'] = X_train['Alternative Dispute Resolution'].replace({'N': 0, 'Y': 1, 'U': 1})
        X_val['Alternative Dispute Resolution Bin'] = X_val['Alternative Dispute Resolution'].replace({'N': 0, 'Y': 1, 'U': 1})
        test['Alternative Dispute Resolution Bin'] = test['Alternative Dispute Resolution'].replace({'N': 0, 'Y': 1, 'U': 1})

        X_train['Attorney/Representative Bin'] = X_train['Attorney/Representative'].replace({'N': 0, 'Y': 1})
        X_val['Attorney/Representative Bin'] = X_val['Attorney/Representative'].replace({'N': 0, 'Y': 1})
        test['Attorney/Representative Bin'] = test['Attorney/Representative'].replace({'N': 0, 'Y': 1})

        train_carriers = set(X_train['Carrier Name'].unique())
        test_carriers = set(test['Carrier Name'].unique())
        common_categories = train_carriers.intersection(test_carriers)
        common_category_map = {category: idx + 1 for idx, 
                           category in enumerate(common_categories)}

        X_train['Carrier Name Enc'] = X_train['Carrier Name'].map(common_category_map).fillna(0).astype(int)
        X_val['Carrier Name Enc'] = X_val['Carrier Name'].map(common_category_map).fillna(0).astype(int)
        test['Carrier Name Enc'] = test['Carrier Name'].map(common_category_map).fillna(0).astype(int)

        X_train, X_val, test = p.encode(X_train, X_val, test, 'Carrier Name Enc', enc)

        X_train, X_val, test = p.encode(X_train, X_val, test, 'Carrier Type', enc)
        X_train, X_val, test = p.encode(X_train, X_val, test, 'Carrier Type', 'OHE')

        X_train, X_val, test = p.encode(X_train, X_val, test, 'County of Injury', enc)

        X_train['COVID-19 Indicator Enc'] = X_train['COVID-19 Indicator'].replace({'N': 0, 'Y': 1})
        X_val['COVID-19 Indicator Enc'] = X_val['COVID-19 Indicator'].replace({'N': 0, 'Y': 1})
        test['COVID-19 Indicator Enc'] = test['COVID-19 Indicator'].replace({'N': 0, 'Y': 1})

        X_train, X_val, test = p.encode(X_train, X_val, test, 'District Name', enc)

        X_train, X_val, test = p.encode(X_train, X_val, test, 'Gender', 'OHE')

        X_train, X_val, test = p.encode(X_train, X_val, test, 'Medical Fee Region', enc)

        X_train, X_val, test = p.encode(X_train, X_val, test, 'Industry Sector', enc)

        drop = ['Alternative Dispute Resolution', 'Attorney/Representative', 'Carrier Type', 'County of Injury',
                'COVID-19 Indicator', 'District Name', 'Gender', 'Carrier Name',
                'Medical Fee Region', 'Industry Sector']

        X_train.drop(columns = drop, axis = 1, inplace = True)
        X_val.drop(columns = drop, axis = 1, inplace = True)
        test.drop(columns = drop, axis = 1, inplace = True)

        # MISSING VALUES
        X_train['C-3 Date Binary'] = X_train['C-3 Date'].notna().astype(int)
        X_val['C-3 Date Binary'] = X_val['C-3 Date'].notna().astype(int)
        test['C-3 Date Binary'] = test['C-3 Date'].notna().astype(int)

        X_train['First Hearing Date Binary'] = X_train['First Hearing Date'].notna().astype(int)
        X_val['First Hearing Date Binary'] = X_val['First Hearing Date'].notna().astype(int)
        test['First Hearing Date Binary'] = test['First Hearing Date'].notna().astype(int)

        drop = ['C-3 Date', 'First Hearing Date']
        X_train.drop(columns = drop, axis = 1, inplace = True)
        X_val.drop(columns = drop, axis = 1, inplace = True)
        test.drop(columns = drop, axis = 1, inplace = True)

        X_train['IME-4 Count'] = X_train['IME-4 Count'].fillna(0)
        X_val['IME-4 Count'] = X_val['IME-4 Count'].fillna(0)
        test['IME-4 Count'] = test['IME-4 Count'].fillna(0)

        X_train['Industry Code'] = X_train['Industry Code'].fillna(0)
        X_val['Industry Code'] = X_val['Industry Code'].fillna(0)
        test['Industry Code'] = test['Industry Code'].fillna(0)

        p.fill_dates(X_train, [X_val, test], 'Accident Date')
        p.fill_dates(X_train, [X_val, test], 'C-2 Date')

        p.fill_dow([X_train, X_val, test], 'Accident Date')
        p.fill_dow([X_train, X_val, test], 'C-2 Date')

        X_train = p.fill_missing_times(X_train, ['Accident to Assembly Time', 
                                 'Assembly to C-2 Time',
                                 'Accident to C-2 Time'])

        X_val = p.fill_missing_times(X_val, ['Accident to Assembly Time', 
                                 'Assembly to C-2 Time',
                                 'Accident to C-2 Time'])

        test = p.fill_missing_times(test, ['Accident to Assembly Time', 
                                 'Assembly to C-2 Time',
                                 'Accident to C-2 Time'])

        p.fill_birth_year([X_train, X_val, test])


        # Variables
        num = ['Age at Injury', 'Average Weekly Wage', 'Birth Year',
           'IME-4 Count', 'Number of Dependents', 'Accident Date Year',
           'Accident Date Month', 'Accident Date Day', 
           'Assembly Date Year', 'Assembly Date Month', 
           'Assembly Date Day', 'C-2 Date Year', 'C-2 Date Month',
           'C-2 Date Day', 'Accident to Assembly Time',
           'Assembly to C-2 Time', 'Accident to C-2 Time']
          # 'Wage to Age Ratio', 'Average Weekly Wage Sqrt',
          # 'IME-4 Count Log', 'IME-4 Count Double Log']


        categ = [var for var in X_train.columns if var not in num]

        categ_count_encoding = ['Carrier Name Enc', 'Carrier Type Enc',
                                'County of Injury Enc', 'District Name Enc',
                                'Medical Fee Region Enc', 
                                'Industry Sector Enc']


        categ_label_bin = [var for var in X_train.columns if var
                           in categ and var not in categ_count_encoding]

        num_count_enc = num + categ_count_encoding


        # Scale
        robust = RobustScaler()

        X_train_num_count_enc_RS = robust.fit_transform(X_train[num_count_enc])
        X_train_num_count_enc_RS = pd.DataFrame(X_train_num_count_enc_RS, columns=num_count_enc, index=X_train.index)
        X_val_num_count_enc_RS = robust.transform(X_val[num_count_enc])
        X_val_num_count_enc_RS = pd.DataFrame(X_val_num_count_enc_RS, columns=num_count_enc, index=X_val.index)
        test_num_count_enc_RS = robust.transform(test[num_count_enc])
        test_num_count_enc_RS = pd.DataFrame(test_num_count_enc_RS, columns=num_count_enc, index=test.index)

        X_train_RS = pd.concat([X_train_num_count_enc_RS, 
                                X_train[categ_label_bin]], axis=1)
        X_val_RS = pd.concat([X_val_num_count_enc_RS, 
                              X_val[categ_label_bin]], axis=1)
        test_RS = pd.concat([test_num_count_enc_RS, 
                             test[categ_label_bin]], axis=1)

        p.ball_tree_impute([X_train_RS, X_val_RS, test_RS], 
                           'Average Weekly Wage')
        
        if outliers:
            X_train_RS = X_train_RS[X_train_RS['Age at Injury'] < 2.0217391304347827]
            
            X_train_RS['Average Weekly Wage Sqrt'] = np.sqrt(X_train_RS['Average Weekly Wage'])

            X_val_RS['Average Weekly Wage Sqrt'] = np.sqrt(X_val_RS['Average Weekly Wage'])

            test_RS['Average Weekly Wage Sqrt'] = np.sqrt(test_RS['Average Weekly Wage'])
            
            upper_limit = X_train_RS['Average Weekly Wage'].quantile(0.99)
            lower_limit = X_train_RS['Average Weekly Wage'].quantile(0.01)

            X_train_RS['Average Weekly Wage'] = X_train_RS['Average Weekly Wage'].clip(lower = lower_limit
                                                                  , upper=upper_limit)
            
            X_train_RS = X_train_RS[X_train_RS['Birth Year'] > -1.9782608695652173]
            
            X_train_RS['IME-4 Count Log'] = np.log1p(X_train_RS['IME-4 Count'])
            X_train_RS['IME-4 Count Double Log'] = np.log1p(X_train_RS['IME-4 Count Log'])

            X_val_RS['IME-4 Count Log'] = np.log1p(X_val_RS['IME-4 Count'])
            X_val_RS['IME-4 Count Double Log'] = np.log1p(X_val_RS['IME-4 Count Log'])

            test_RS['IME-4 Count Log'] = np.log1p(test_RS['IME-4 Count'])
            test_RS['IME-4 Count Double Log'] = np.log1p(test_RS['IME-4 Count Log'])
            
            X_train_RS = X_train_RS[X_train_RS['Accident Date Year'] > -2.0]
            
            X_train_RS = X_train_RS[X_train_RS['C-2 Date Year'] > -2.0]
            
            y_train = y_train[X_train_RS.index]

        # Training
        model = run_model(model_name, X_train_RS, y_train, params.get(model_name, {}))

        # Predictions
        pred_train = model.predict(X_train_RS)
        pred_val = model.predict(X_val_RS)
        
        test_preds += model.predict_proba(test_RS)

        # Metrics
        f1macro_train.append(f1_score(y_train, pred_train, average='macro'))
        f1macro_val.append(f1_score(y_val, pred_val, average='macro'))
        precision_train.append(precision_score(y_train, pred_train, average='macro')) 
        precision_val.append(precision_score(y_val, pred_val, average='macro'))  
        recall_train.append(recall_score(y_train, pred_train, average='macro'))
        recall_val.append(recall_score(y_val, pred_val, average='macro'))
        
        # Compute Time
        end_time = time.time()
        elapsed_time = round((end_time - start_time) / 60, 2)
        timer.append(elapsed_time) 
        print(f'This Fold took {elapsed_time} minutes')

    # Metrics Average and Stdev
    avg_time = round(np.mean(timer), 3)
    avg_f1_train = round(np.mean(f1macro_train), 3)
    avg_f1_val = round(np.mean(f1macro_val), 3)
    avg_precision_train = round(np.mean(precision_train), 3)
    avg_precision_val = round(np.mean(precision_val), 3)
    avg_recall_train = round(np.mean(recall_train), 3)
    avg_recall_val = round(np.mean(recall_val), 3)
    std_time = round(np.std(timer), 3)
    std_f1_train = round(np.std(f1macro_train), 3)
    std_f1_val = round(np.std(f1macro_val), 3)
    std_precision_train = round(np.std(precision_train), 3)
    std_precision_val = round(np.std(precision_val), 3)
    std_recall_train = round(np.std(recall_train), 3)
    std_recall_val = round(np.std(recall_val), 3)

    # Final Predictions using Soft Voting
    final_test_preds = np.argmax(test_preds / method.get_n_splits(), axis=1)
    test_RS['Claim Injury Type'] = final_test_preds 

    test_RS['Claim Injury Type'] = test_RS['Claim Injury Type'].replace(label_mapping)
    
    predictions = test_RS['Claim Injury Type']
    
    if file_name != None:
    
        predictions.to_csv(f'./pred/{file_name}.csv')


    # Return data and treated Test_RS
    return {
        'avg_time': str(avg_time) + '+/-' + str(std_time),
        'avg_f1_train': str(avg_f1_train) + '+/-' + str(std_f1_train),
        'avg_f1_val': str(avg_f1_val) + '+/-' + str(std_f1_val),
        'avg_precision_train': str(avg_precision_train) + '+/-' + str(std_precision_train),
        'avg_precision_val': str(avg_precision_val) + '+/-' + str(std_precision_val),
        'avg_recall_train': str(avg_recall_train) + '+/-' + str(std_recall_train),
        'avg_recall_val': str(avg_recall_val) + '+/-' + str(std_recall_val),
        'test_data': test_RS,
        'predictions': predictions
    }