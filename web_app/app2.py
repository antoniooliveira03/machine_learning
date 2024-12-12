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
from predict import show_predict



# Load the logo
img = Image.open("24_Nexus_Analytics.png")
logo = Image.open("Nova_IMS.png")

# Define the navigation menu
def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Inputs and Prediction", "Explore Data"],
        icons=["house", "check-circle", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    return selected


selected = streamlit_menu()

st.logo(
    logo,
    size="large")

# Home Page
if selected == "Home":
    st.title("Grant Compensation Benefit Prediction App")

    # Dropdown for team members
    st.subheader("Project Team")
    with st.expander("Click to view team details"):
        team_members = {
            "Data Scientist Manager": ("António Oliveira", "20211595"),
            "Data Scientist Senior": ("Tomás Ribeiro", "20240526"),
            "Data Scientist Junior": ("Gonçalo Pacheco", "20240695"),
            "Data Analyst Senior": ("Gonçalo Custódio", "20211643"),
            "Data Analyst Junior": ("Ana Caleiro", "20240696"),
        }
        for role, (name, student_id) in team_members.items():
            st.write(f"**{role}**: {name} ({student_id})")

    # App description
    st.markdown("""
    This application is designed to assist the New York Workers' Compensation Board (WCB) in automating the classification of compensation claims. It uses a machine learning model trained on historical claims data from 2020 to 2022 to predict whether a compensation benefit will be granted based on various claim-related features.
    To use the application, the WCB simply needs to complete a quick form on the Inputs and Predictions page. Additionally, we provide an Explore Data page, which allows the board to analyze the data powering the model for more specific insights.
    """)

    #Inputing the metadata 
    st.subheader("Metadata")
    with st.expander("Click to view detailed information about the questions in the form"):
        metadata = {
            "County of Injury": "Name of the New York County where the injury occurred.",
            "COVID-19 Indicator": "Indication that the claim may be associated with COVID-19.",
            "District Name": "Name of the WCB district office that oversees claims for that region or area of the state.",
            "First Hearing Date": "Date the first hearing was held on a claim at a WCB hearing location. A blank date means the claim has not yet had a hearing held.",
            "Gender": "The reported gender of the injured worker.",
            "IME-4 Count": "Number of IME-4 forms received per claim. The IME-4 form is the 'Independent Examiner's Report of Independent Medical Examination' form.",
            "Industry Code": "NAICS code and descriptions are available at: https://www.naics.com/search-naics-codes-by-industry/.",
            "Industry Code Description": "2-digit NAICS industry code description used to classify businesses according to their economic activity.",
            "Medical Fee Region": "Approximate region where the injured worker would receive medical service.",
            "OIICS Nature of Injury Description": "The OIICS nature of injury codes & descriptions are available at https://www.bls.gov/iif/oiics_manual_2007.pdf.",
            "WCIO Cause of Injury Code": "The WCIO cause of injury codes & descriptions are at https://www.wcio.org/Active%20PNC/WCIO_Cause_Table.pdf.",
            "WCIO Cause of Injury Description": "See description of field above.",
            "WCIO Nature of Injury Code": "The WCIO nature of injury codes are available at https://www.wcio.org/Active%20PNC/WCIO_Nature_Table.pdf.",
            "WCIO Nature of Injury Description": "See description of field above.",
            "WCIO Part Of Body Code": "The WCIO part of body codes & descriptions are available at https://www.wcio.org/Active%20PNC/WCIO_Part_Table.pdf.",
            "WCIO Part Of Body Description": "See description of field above.",
            "Zip Code": "The reported ZIP code of the injured worker's home address.",
            "Agreement Reached": "Binary variable: Yes if there is an agreement without the involvement of the WCB -> unknown at the start of a claim.",
            "WCB Decision": "Multiclass variable: Decision of the WCB relative to the claim: 'Accident' means that claim refers to workplace accident, 'Occupational Disease' means illness from the workplace. -> requires WCB deliberation so it is unknown at start of claim.",
            "Claim Injury Type": "Main variable: Deliberation of the WCB relative to benefits awarded to the claim. Numbering indicates severity."
        }

        # Transform the Dict in a pandas data frame
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Attribute", "Description"])
        
        # Display the table
        st.dataframe(metadata_df)

    # Display the logo below the team section
    st.image(img, use_container_width=True)

#Inputs and Predictions Page 
if selected == "Inputs and Prediction":
    st.title("Predict Compensation Benefit")
    st.markdown(
        "Provide the necessary details below to predict whether a compensation benefit will be granted."
    )

    # Run the function of Predictions
    show_predict()

#Explore Data Page 
if selected == "Explore Data":
    st.title("Model Data and Insights")
    st.markdown("View model's performance, confusion matrix, and more.")

    #Run the Function of Explore Data 