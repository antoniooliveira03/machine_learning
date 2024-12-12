import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import preproc as p

# Load the logo
#img = Image.open("24_Nexus_Analytics.png")

### TEMP
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
    This application is designed to assist the New York Workers' Compensation Board (WCB) in automating the classification of compensation claims. It uses placeholders to simulate data input and predictions.
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

    # Display user inputs
    st.subheader("Your Inputs")
    input_data = {
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

    input_df = pd.DataFrame(input_data, index=[0])

    st.write(input_df)

    # Placeholder for prediction button
    st.subheader("Prediction")
    if st.button("Predict"):
        csv_path = "user_inputs.csv"
        input_df.to_csv(csv_path, index=False)
        st.success(f"User inputs saved to {csv_path}")

        # Call the preprocessing and prediction function
        prediction = p.preproc_(csv_path)
        st.subheader("Prediction Result")
        st.write(f"The predicted compensation benefit is: {prediction}")


# Model Data Page
if selected == "Model Data":
    st.title("Model Data and Insights")
    st.markdown("View model's performance, confusion matrix, and more.")
    st.info("No model data available. This page is a placeholder.")
