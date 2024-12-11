import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pickle
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

# Loading the Model
#def load_model():
    #with gzip.open("trained_model.pkl.gz", "rb") as file:
        #data = pickle.load(file)
    #return data

#model = load_model()

# Starting the Encoding Methods 



def show_predict():
    birth_year = st.slider("Worker Birth Year",1944, 2024, 1944)



