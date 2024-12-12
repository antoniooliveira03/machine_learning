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
                2020,
                1980,
                help="Year of the worker's birth.",
            )

            gender = st.radio(
                "Gender",
                ["M", "F", 'U/X'],
                horizontal=True,
                help="Indicate the gender of the worker.",
            )

        with col2:
            avg_weekly_wage = st.number_input(
                "Average Weekly Wage (USD)",
                min_value=0,
                value=500,
                help="Enter the average weekly wage of the worker.",
            )

            zip_code = st.slider(
                "Zip Code",
                0,
                100,
                2,
                help="Zip code of the worker.",
            )

            n_dependents = st.slider(
                "Number of Dependents",
                0,
                100,
                2,
                help="Number of dependents of the worker.",
            )

    with st.expander("Incident Details"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            accident_year = st.slider(
                "Accident Year", 1966, 2021, 2015, help="Year of the accident."
            )
            accident_month = st.slider(
                "Accident Month", 1, 12, 6, help="Month of the accident."
            )

            accident_day = st.slider(
                "Accident Day", 1, 31, 6, help="Day of the accident."
            )

            part_of_body = st.selectbox(
                "Part of Body Injured",
                list(part_of_body_mapping.keys()),
                help="Select the injured body part.",
            )

        with col2:
            assembly_year = st.slider(
                "Assembly Year", 1966, 2021, 2015, help="Year of the assembly."
            )
            assembly_month = st.slider(
                "Assembly Month", 1, 12, 6, help="Month of the assembly."
            )

            assembly_day = st.slider(
                "Assembly Day", 1, 31, 6, help="Day of the assembly."
            )

            cause_injury = st.selectbox(
                "Cause of Injury",
                list(cause_of_injury_mapping.keys()),
                help="Select the cause of injury.",
            )

        with col3:
            c2_year = st.slider(
                "C-2 Year", 1966, 2021, 2015, help="Year of the C-2."
            )
            c2_month = st.slider(
                "C-2 Month", 1, 12, 6, help="Month of the C-2."
            )

            c2_day = st.slider(
                "C-2 Day", 1, 31, 6, help="Day of the C-2."
            )

            covid = st.radio(
                "Covid-19?",
                ["Yes", "No"],
                horizontal=True,
                help="Indicate if the claim is related to Covid-19.",
            )

        with col4:
            c3_year = st.slider(
                "C-3 Year", 1966, 2021, 2015, help="Year of the C-3."
            )
            c3_month = st.slider(
                "C-3 Month", 1, 12, 6, help="Month of the C-3."
            )

            c3_day = st.slider(
                "C-3 Day", 1, 31, 6, help="Day of the C-3."
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

            carrier_type = st.selectbox(
                "Carrier Type",
                [
                    "1"
                ],
                help="Select the carrier type.",
            )

            county = st.selectbox(
                "County of Injury",
                [
                    "1"
                ],
                help="Select the county where the injury occurred.",
            )

            district = st.selectbox(
                "District Name",
                [
                    "1"
                ],
                help="Select the district of the incident.",
            )


            attorney = st.radio(
                "Claim Represented by Attorney?",
                ["Yes", "No"],
                horizontal=True,
                help="Indicate if the claim is represented by an attorney.",
            )

            alternative_dispute = st.radio(
                "Alternative Dispute Resolution?",
                ["Yes", "No"],
                horizontal=True,
                help="Indicate if alternative dispute resolution is applicable.",
            )
        with col2:
            first_hearing_year = st.slider(
                "First Hearing Year",
                2020,
                2024,
                2022,
                help="Year of the first hearing.",
            )

            first_hearing_month = st.slider(
                "First Hearing Month",
                1,
                12,
                3,
                help="Month of the first hearing.",
            )

            first_hearing_day = st.slider(
                "First Hearing Day",
                1,
                31,
                12,
                help="Day of the first hearing.",
            )
            ime_4_count = st.number_input(
                "IME-4 Forms Received Count",
                min_value=0,
                max_value=30,
                value=1,
                help="Number of IME-4 forms received.",
            )

    # Additional Inputs
    with st.expander("Additional Information"):
        C_2_day = st.slider(
            "In which day did you receive the Employer's Report of Work-Related Injury/Illness",
            1,
            31,
            1,
        )

        indust_code_descript = st.selectbox(
            "Industry Description",
            ["1"],
            help = 'None'
        )

        cause = st.selectbox(
            "Cause of Injury Description",
            ["1"],
            help = 'None'
        )

        nature_description = st.selectbox(
            "Nature of Injury Description",
            ["1"],
            help = 'None'
        )


    # Collecting all inputs into the input_df dictionary
    st.subheader("Your Inputs")
    input_data = {
        "Birth Year": birth_year,
        "Gender": gender,
        "Average Weekly Wage": avg_weekly_wage,
        "Zip Code": zip_code,
        "Number of Dependents": n_dependents,
        "Accident Year": accident_year,
        "Accident Month": accident_month,
        "Accident Day": accident_day,
        "Part of Body Injured": part_of_body,
        "Cause of Injury": cause_injury,
        "Assembly Year": assembly_year,
        "Assembly Month": assembly_month,
        "Assembly Day": assembly_day,
        "C-2 Year": c2_year,
        "C-2 Month": c2_month,
        "C-2 Day": c2_day,
        "C-3 Year": c3_year,
        "C-3 Month": c3_month,
        "C-3 Day": c3_day,
        "Covid": covid,
        "Carrier Name": carrier_name,
        "Carrier Type": carrier_type,
        "County": county,
        "District": district,
        "Attorney/Representative Bin": 1 if attorney == "Yes" else 0,
        "Alternative Dispute Resolution": 1 if alternative_dispute == "Yes" else 0,
        "First Hearing Year": first_hearing_year,
        "First Hearing Month": first_hearing_month,
        "First Hearing Day": first_hearing_day,
        "IME-4 Count": ime_4_count,
        "C-2 Day": C_2_day,
        "Industry Description": indust_code_descript,
        "Cause Description": cause,
        "Nature Description": nature_description,
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
