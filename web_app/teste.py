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
def load_model():
    with gzip.open(
        "/Users/goncalopacheco/Documents/GitHub/machine_learning/web_app/trained_model.pkl.gz",
        "rb",
    ) as file:
        data = pickle.load(file)
    return data


model = load_model()

# Nature of Injury Mapping
nature_of_injury_mapping = {
    "CONTUSION": 1,
    "SPRAIN OR TEAR": 2,
    "CONCUSSION": 3,
    "PUNCTURE": 4,
    "LACERATION": 5,
    "ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC": 6,
    "ALL OTHER SPECIFIC INJURIES, NOC": 7,
    "BURN": 8,
    "STRAIN OR TEAR": 9,
    "FRACTURE": 10,
    "INFLAMMATION": 11,
    "FOREIGN BODY": 12,
    "MULTIPLE PHYSICAL INJURIES ONLY": 13,
    "RUPTURE": 14,
    "DISLOCATION": 15,
    "ALL OTHER CUMULATIVE INJURY, NOC": 16,
    "HERNIA": 17,
    "ANGINA PECTORIS": 18,
    "CARPAL TUNNEL SYNDROME": 19,
    "NO PHYSICAL INJURY": 20,
    "INFECTION": 21,
    "CRUSHING": 22,
    "SYNCOPE": 23,
    "POISONING - GENERAL (NOT OD OR CUMULATIVE)": 24,
    "RESPIRATORY DISORDERS": 25,
    "HEARING LOSS OR IMPAIRMENT": 26,
    "MENTAL STRESS": 27,
    "SEVERANCE": 28,
    "ELECTRIC SHOCK": 29,
    "LOSS OF HEARING": 30,
    "DUST DISEASE, NOC": 31,
    "DERMATITIS": 32,
    "ASPHYXIATION": 33,
    "MENTAL DISORDER": 34,
    "CONTAGIOUS DISEASE": 35,
    "AMPUTATION": 36,
    "MYOCARDIAL INFARCTION": 37,
    "POISONING - CHEMICAL, (OTHER THAN METALS)": 38,
    "MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL": 39,
    "VISION LOSS": 40,
    "VASCULAR": 41,
    "COVID-19": 42,
    "CANCER": 43,
    "HEAT PROSTRATION": 44,
    "AIDS": 45,
    "ASBESTOSIS": 46,
    "POISONING - METAL": 47,
    "VDT - RELATED DISEASES": 48,
    "FREEZING": 49,
    "BLACK LUNG": 50,
    "SILICOSIS": 51,
    "ADVERSE REACTION TO A VACCINATION OR INOCULATION": 52,
    "HEPATITIS C": 53,
    "RADIATION": 54,
    "ENUCLEATION": 55,
    "BYSSINOSIS": 56,
}

# Encoding for "WCIO Cause of Injury Code"
cause_of_injury_mapping = {
    "FROM LIQUID OR GREASE SPILLS": 1,
    "REPETITIVE MOTION": 2,
    "OBJECT BEING LIFTED OR HANDLED": 3,
    "HAND TOOL, UTENSIL; NOT POWERED": 4,
    "FALL, SLIP OR TRIP, NOC": 5,
    "CUT, PUNCTURE, SCRAPE, NOC": 6,
    "OTHER - MISCELLANEOUS, NOC": 7,
    "STRUCK OR INJURED, NOC": 8,
    "CHEMICALS": 9,
    "COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE": 10,
    "LIFTING": 11,
    "TWISTING": 12,
    "ON SAME LEVEL": 13,
    "STRAIN OR INJURY BY, NOC": 14,
    "MOTOR VEHICLE, NOC": 15,
    "FROM DIFFERENT LEVEL (ELEVATION)": 16,
    "PUSHING OR PULLING": 17,
    "FOREIGN MATTER (BODY) IN EYE(S)": 18,
    "FALLING OR FLYING OBJECT": 19,
    "FELLOW WORKER, PATIENT OR OTHER PERSON": 20,
    "STATIONARY OBJECT": 21,
    "ON ICE OR SNOW": 22,
    "ABSORPTION, INGESTION OR INHALATION, NOC": 23,
    "PERSON IN ACT OF A CRIME": 24,
    "ON STAIRS": 25,
    "FROM LADDER OR SCAFFOLDING": 26,
    "SLIP, OR TRIP, DID NOT FALL": 27,
    "JUMPING OR LEAPING": 28,
    "MOTOR VEHICLE": 29,
    "RUBBED OR ABRADED, NOC": 30,
    "REACHING": 31,
    "OBJECT HANDLED": 32,
    "ELECTRICAL CURRENT": 33,
    "HOLDING OR CARRYING": 34,
    "CAUGHT IN, UNDER OR BETWEEN, NOC": 35,
    "FIRE OR FLAME": 36,
    "HOT OBJECTS OR SUBSTANCES": 37,
    "POWERED HAND TOOL, APPLIANCE": 38,
}

# Encoding for "Part of Body Injured"
part_of_body_mapping = {
    "BUTTOCKS": 1,
    "SHOULDER(S)": 2,
    "MULTIPLE HEAD INJURY": 3,
    "FINGER(S)": 4,
    "LUNGS": 5,
    "EYE(S)": 6,
    "ANKLE": 7,
    "THUMB": 8,
    "LOWER BACK AREA": 9,
    "ABDOMEN INCLUDING GROIN": 10,
    "LOWER LEG": 11,
    "HIP": 12,
    "UPPER LEG": 13,
    "MOUTH": 14,
    "KNEE": 15,
    "WRIST": 16,
    "SPINAL CORD": 17,
    "HAND": 18,
    "SOFT TISSUE": 19,
    "UPPER ARM": 20,
    "ELBOW": 21,
    "MULTIPLE UPPER EXTREMITIES": 22,
    "MULTIPLE BODY PARTS (INCLUDING BODY": 23,
    "BODY SYSTEMS AND MULTIPLE BODY SYSTEMS": 24,
    "MULTIPLE NECK INJURY": 25,
    "FOOT": 26,
    "EAR(S)": 27,
    "MULTIPLE LOWER EXTREMITIES": 28,
    "DISC": 29,
    "LOWER ARM": 30,
    "MULTIPLE": 31,
    "CHEST": 32,
    "UPPER BACK AREA": 33,
    "SKULL": 34,
    "WRIST (S) & HAND(S)": 35,
    "TOES": 36,
    "FACIAL BONES": 37,
    "NO PHYSICAL INJURY": 38,
    "MULTIPLE TRUNK": 39,
    "WHOLE BODY": 40,
    "INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED": 41,
    "PELVIS": 42,
    "NOSE": 43,
    "GREAT TOE": 44,
    "INTERNAL ORGANS": 45,
    "VERTEBRAE": 46,
    "LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA": 47,
    "BRAIN": 48,
    "SACRUM AND COCCYX": 49,
    "ARTIFICIAL APPLIANCE": 50,
    "TEETH": 51,
    "LARYNX": 52,
    "HEART": 53,
    "TRACHEA": 54,
}

# Industry Code Mapping
industry_code_mapping = {
    "RETAIL TRADE": 1,
    "CONSTRUCTION": 2,
    "ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT": 3,
    "HEALTH CARE AND SOCIAL ASSISTANCE": 4,
    "ACCOMMODATION AND FOOD SERVICES": 5,
    "EDUCATIONAL SERVICES": 6,
    "MANUFACTURING": 7,
    "TRANSPORTATION AND WAREHOUSING": 8,
    "WHOLESALE TRADE": 9,
    "REAL ESTATE AND RENTAL AND LEASING": 10,
    "FINANCE AND INSURANCE": 11,
    "OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)": 12,
    "PUBLIC ADMINISTRATION": 13,
    "INFORMATION": 14,
    "PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES": 15,
    "ARTS, ENTERTAINMENT, AND RECREATION": 16,
    "UTILITIES": 17,
    "MINING": 18,
    "AGRICULTURE, FORESTRY, FISHING AND HUNTING": 19,
    "MANAGEMENT OF COMPANIES AND ENTERPRISES": 20,
}


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


# Encoding 'Yes'/'No' values for C_3_date
def encode_binary_value(value):
    return 1 if value == "Yes" else 0


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
        nature_of_injury = st.selectbox(
            "Nature of Injury",
            list(nature_of_injury_mapping.keys()),
            help="Select the nature of injury.",
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

# Additional Inputs
with st.expander("Additional Information"):
    C_2_day = st.slider(
        "In which day did you receive the Employer's Report of Work-Related Injury/Illness",
        1,
        31,
        1,
    )
    C_3_date = st.selectbox("Did the Employee Claim Form was received?", ["Yes", "No"])
    indust_code_descript = st.selectbox(
        "Industry Description",
        list(industry_code_mapping.keys()),
    )

# Prepare the input data for the model

input_data = [
    encode_carrier_name(carrier_name),
    birth_year,
    avg_weekly_wage,
    accident_year,
    accident_month,
    part_of_body_mapping.get(part_of_body, 0),
    cause_of_injury_mapping.get(cause_injury, 0),
    nature_of_injury_mapping.get(nature_of_injury, 0),
    encode_binary_value(attorney),
    first_hearing_year,
    ime_4_count,
    C_2_day,
    encode_binary_value(C_3_date),
    industry_code_mapping.get(indust_code_descript, 0),
]


# Prediction
st.subheader("Prediction")
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

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

        predicted_class = np.argmax(prediction_proba)
        predicted_label = claim_injury_type_mapping.get(
            predicted_class, "Unknown Class"
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
        st.write(
            f"Prediction: **{predicted_label}** with confidence **{prediction_proba[predicted_class] * 100:.2f}%**."
        )

    except ValueError as e:
        st.error(f"Prediction failed: {e}")
