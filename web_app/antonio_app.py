import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np

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
    This application is designed to assist the New York Workers' Compensation Board (WCB) in automating the classification of compensation claims. It uses placeholders to simulate data input and predictions.
    """)

    # Display the logo below the team section
    st.image(img, use_container_width=True)

# Inputs and Prediction Page
if selected == "Inputs and Prediction":
    st.title("Predict Compensation Benefit")
    st.markdown(
        "Provide the necessary details below to simulate a prediction."
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
                ["Hand", "Back", "Foot", "Head", "Other"],
                help="Select the injured body part.",
            )
            cause_injury = st.selectbox(
                "Cause of Injury",
                ["Liquid Spills", "Repetitive Motion", "Lifting", "Other"],
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

    # Display user inputs
    st.subheader("Your Inputs")
    input_data = {
        "Birth Year": birth_year,
        "Average Weekly Wage": avg_weekly_wage,
        "Accident Year": accident_year,
        "Accident Month": accident_month,
        "Part of Body Injured": part_of_body,
        "Cause of Injury": cause_injury,
        "Carrier Name": carrier_name,
        "Attorney Representation": attorney,
        "First Hearing Year": first_hearing_year,
        "IME-4 Count": ime_4_count,
    }
    st.write(pd.DataFrame([input_data]))

    # Placeholder for prediction button
    st.subheader("Prediction")
    if st.button("Predict"):
        st.info("Prediction logic is not implemented yet. This is a placeholder.")

# Model Data Page
if selected == "Model Data":
    st.title("Model Data and Insights")
    st.markdown("View model's performance, confusion matrix, and more.")
    st.info("No model data available. This page is a placeholder.")
