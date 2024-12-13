import pandas as pd
import utils as u
import streamlit as st
import datetime


def preproc_(path):
    
    user_input = pd.read_csv(path)

    st.text("Processing your input data...")

    user_input['Age at Injury'] = 2024 - user_input['Birth Year']

    # List of columns to convert to datetime
    date_columns = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']

    # Apply pd.to_datetime() to each column in the list for both df and test
    for col in date_columns:
        user_input[col] = pd.to_datetime(user_input[col], errors='coerce')

    mapping = {
    '5D. SPECIAL FUND - UNKNOWN': '5. SPECIAL FUND OR UNKNOWN',
    '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)': '5. SPECIAL FUND OR UNKNOWN',
    '5C. SPECIAL FUND - POI CARRIER WCB MENANDS': '5. SPECIAL FUND OR UNKNOWN',
    'UNKNOWN': '5. SPECIAL FUND OR UNKNOWN'}  

    user_input['Carrier Type'] = user_input['Carrier Type'].replace(mapping)

    mapping = {  
        'M': 'M',
        'F': 'F',
        'U': 'U/X',  
        'X': 'U/X' }

    user_input['Gender'] = user_input['Gender'].map(mapping)  

    user_input['Gender Enc'] = user_input['Gender'].replace({'M': 0, 'F': 1, 'U/X': 2})


    for column in user_input.columns:
        # Check if the column is a datetime type
        if pd.api.types.is_datetime64_any_dtype(user_input[column]) and column not in ['C-3 Date', 'First Hearing Date']:
            user_input[f'{column} Day of Week'] = user_input[column].dt.weekday 

    user_input['Accident to Assembly Time'] = (user_input['Assembly Date'] - user_input['Accident Date']).dt.days

    user_input['Assembly to C-2 Time'] = (user_input['Assembly Date'] - user_input['C-2 Date']).dt.days

    user_input['Accident to C-2 Time'] = (user_input['C-2 Date'] - user_input['Accident Date']).dt.days

    user_input['WCIO Part Of Body Code'] = user_input['WCIO Part Of Body Code'].abs()

    columns_to_join = [
    'WCIO Cause of Injury Code',
    'WCIO Nature of Injury Code',
    'WCIO Part Of Body Code'
    ]

    user_input[columns_to_join] = user_input[columns_to_join].fillna(0).astype(int)

    user_input['WCIO Codes'] = user_input[columns_to_join].astype(str).agg(''.join, axis=1).astype(int)

    user_input['Insurance'] = user_input['Carrier Name'].str.contains('ins', case=False, na=False).astype(int)

    # Create a new column 'Zip Code Valid' to flag the validity of the 'Zip Code' field
    user_input['Zip Code Valid'] = user_input['Zip Code'].apply(
        lambda x: 2 if pd.isna(x)          
        else (1 if not str(x).isnumeric()   
            else 0)                       
    )

    user_input['Industry Sector'] = user_input['Industry Code Description'].apply(u.group_industry)

    bins = [-1, 17, 64, 117]
    labels = [0, 1, 2]

    user_input['Age Group'] = pd.cut(user_input['Age at Injury'], 
                                    bins=bins, labels=labels, right=True)
    
    drop = ['Accident Date', 'Assembly Date',
        'C-2 Date', 'Zip Code']
    

    user_input.drop(columns = drop, axis = 1, inplace = True)

    return 101