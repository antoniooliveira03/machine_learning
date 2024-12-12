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


# Load the pre-trained model
def load_model():
    with gzip.open("trained_model.pkl.gz", "rb") as file:
        model = pickle.load(file)
    return model

#Start the encoding methods 

# Manual frequency encoding for 'Carrier Name'
def encode_carrier_name(carrier_name):
    carrier_name_freq_map = {
        "NEW HAMPSHIRE INSURANCE CO": 0.05,
        "ZURICH AMERICAN INSURANCE CO": 0.10,
        "INDEMNITY INSURANCE CO OF": 0.07,
        "MARATHON CENTRAL SCHOOL DIST": 0.02,
        "CAMBRIDGE CENTRAL SCHOOL": 0.04,
        "HERMON-DEKALB CENTRAL": 0.03,
    }
    return carrier_name_freq_map.get(carrier_name, 0)  # Default to 0 if unknown


# Encoding for "WCIO Cause of Injury Code"
cause_of_injury_mapping = {
    "Liquid Spills": 1,
    "Repetitive Motion": 2,
    "Lifting": 3,
    "Other": 4,
}

# Encoding for "Part of Body Injured"
part_of_body_mapping = {
    "Hand": 1,
    "Back": 2,
    "Foot": 3,
    "Head": 4,
    "Other": 5,
}

# Load model
model = load_model()

# Load the logo
img = Image.open("24_Nexus_Analytics.png")


# Define the navigation menu
def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Inputs and Prediction", "Model Data"],
        icons=["house", "check-circle", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    return selected


selected = streamlit_menu()

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
    """)

    # Display the logo below the team section
    st.image(img, use_container_width=True)

# Inputs and Prediction Page
if selected == "Inputs and Prediction":
    st.title("Predict Compensation Benefit")
    st.markdown(
        "Provide the necessary details below to predict whether a compensation benefit will be granted."
    )

    # Input fields grouped logically
    st.header("Claim Details")
    with st.expander("Personal Information"):
        col1, col2 = st.columns(2)
        with col1:
            birth_year = st.slider(
                "Worker Birth Year",
                1944,
                2006,
                1980,
                help="Year of the worker's birth.",
            )
        with col2:
            avg_weekly_wage = st.number_input(
                "Average Weekly Wage (USD)",
                min_value=0,
                value=500,
                help="Enter the average weekly wage of the worker.",
            )

    with st.expander("Incident Details"):
        col1, col2 = st.columns(2)
        with col1:
            accident_year = st.slider(
                "Accident Year", 1966, 2021, 2015, help="Year of the accident."
            )
            accident_month = st.slider(
                "Accident Month", 1, 12, 6, help="Month of the accident."
            )
        with col2:
            part_of_body = st.selectbox(
                "Part of Body Injured",
                list(part_of_body_mapping.keys()),
                help="Select the injured body part.",
            )
            cause_injury = st.selectbox(
                "Cause of Injury",
                list(cause_of_injury_mapping.keys()),
                help="Select the cause of injury.",
            )

    with st.expander("Administrative Information"):
        col1, col2 = st.columns(2)
        with col1:
            carrier_name = st.selectbox(
                "Carrier Name",
                [
                    "NEW HAMPSHIRE INSURANCE CO",
                    "ZURICH AMERICAN INSURANCE CO",
                    "INDEMNITY INSURANCE CO OF",
                    "MARATHON CENTRAL SCHOOL DIST",
                    "CAMBRIDGE CENTRAL SCHOOL",
                    "HERMON-DEKALB CENTRAL",
                ],
                help="Select the insurance carrier handling the claim.",
            )
            attorney = st.radio(
                "Claim Represented by Attorney?",
                ["Yes", "No"],
                horizontal=True,
                help="Indicate if the claim is represented by an attorney.",
            )
        with col2:
            first_hearing_year = st.slider(
                "First Hearing Year",
                2020,
                2024,
                2022,
                help="Year of the first hearing.",
            )
            ime_4_count = st.number_input(
                "IME-4 Forms Received Count",
                min_value=0,
                max_value=30,
                value=1,
                help="Number of IME-4 forms received.",
            )

    # Manual frequency encoding for 'Carrier Name'
    carrier_name_freq = encode_carrier_name(carrier_name)

    # Encode categorical features
    cause_injury_encoded = cause_of_injury_mapping[cause_injury]
    part_of_body_encoded = part_of_body_mapping[part_of_body]

    # Combine all input data into a dictionary
    data = {
        "Birth Year": birth_year,
        "Average Weekly Wage": avg_weekly_wage,
        "Accident Year": accident_year,
        "Accident Month": accident_month,
        "WCIO Part Of Body Code": part_of_body_encoded,
        "WCIO Cause of Injury Code": cause_injury_encoded,
        "Carrier Name freq": carrier_name_freq,
        "Attorney/Representative Bin": 1 if attorney == "Yes" else 0,
        "First Hearing Year": first_hearing_year,
        "IME-4 Count Log": np.log1p(ime_4_count),
        "C-3 Date Binary": 1,  # Placeholder
        "Industry Code": 1,  # Placeholder
        "WCIO Nature of Injury Code": 1,  # Placeholder
        "C-2 Day": 1,  # Placeholder
    }
    input_df = pd.DataFrame(data, index=[0])

    # Align columns with model features
    expected_features = (
        model.feature_names_in_
    )  # Get feature names from the trained model
    input_df_aligned = input_df.reindex(columns=expected_features, fill_value=0)

    # Display user inputs
    st.subheader("Your Inputs")
    st.write(input_df_aligned)

    # Prediction
    st.subheader("Prediction")
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df_aligned)[0]
            prediction_proba = model.predict_proba(input_df_aligned)[0]

            # Add a dictionary mapping of class indices to meaningful labels
            claim_injury_type_mapping = {
                1: "CANCELLED",
                2: "NON-COMP",
                3: "MED ONLY",
                4: "TEMPORARY",
                5: "PPD SCH LOSS",
                6: "PPD NSL",
                7: "PTD",
                8: "DEATH",
            }

            # Get the predicted class
            predicted_class = np.argmax(
                prediction_proba
            )  # Get the index of the class with the highest probability
            predicted_label = claim_injury_type_mapping.get(
                predicted_class, "Unknown Class"
            )

            # Display the predicted class and its corresponding label with confidence
            st.write(
                f"The predicted Claim Injury Type is: **{predicted_label}** with a confidence of **{prediction_proba[predicted_class] * 100:.2f}%**."
            )

            # Provide a detailed explanation of the prediction
            injury_explanations = {
                1: "The claim was canceled, and no compensation will be provided.",
                2: "The claim is deemed ineligible for compensation under workers' compensation laws.",
                3: "The claim will be treated with medical-only benefits, typically involving medical treatments without compensation for time off work.",
                4: "The worker's condition is considered temporary, and compensation may be granted for the recovery period.",
                5: "The worker has sustained a permanent partial disability with a scheduled loss of function (e.g., loss of a limb).",
                6: "The worker has sustained a permanent partial disability with no scheduled loss but permanent impairment.",
                7: "The worker is permanently and totally disabled, typically resulting in long-term compensation.",
                8: "The claim is related to a fatal injury, and compensation may be granted to the worker’s beneficiaries.",
            }

            # Display injury explanation
            st.write(
                f"Explanation: {injury_explanations.get(predicted_class, 'No explanation available for this type.')}"
            )

            # Optionally, display top 2 predicted classes for user awareness
            top_n = 2
            st.write("Top predictions and their confidence levels:")
            top_n_predictions = sorted(
                zip(claim_injury_type_mapping.values(), prediction_proba),
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]
            for label, prob in top_n_predictions:
                st.write(f"{label}: {prob * 100:.2f}%")
        except ValueError as e:
            st.error(f"Prediction failed: {e}")

# Model Data Page
if selected == "Model Data":
    st.title("Model Data and Insights")
    st.markdown("View model's performance, confusion matrix, and more.")
    cm = confusion_matrix([0, 1, 1, 0], [0, 1, 1, 0])  # Replace with real values
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    st.pyplot()
