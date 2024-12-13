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
    # Create the dictonary 
    cause_injury_descript = ('FROM LIQUID OR GREASE SPILLS', 'REPETITIVE MOTION',
       'OBJECT BEING LIFTED OR HANDLED',
       'HAND TOOL, UTENSIL; NOT POWERED', 'FALL, SLIP OR TRIP, NOC',
       'CUT, PUNCTURE, SCRAPE, NOC', 'OTHER - MISCELLANEOUS, NOC',
       'STRUCK OR INJURED, NOC', 'CHEMICALS',
       'COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE', 'LIFTING',
       'TWISTING', 'ON SAME LEVEL', 'STRAIN OR INJURY BY, NOC',
       'MOTOR VEHICLE, NOC', 'FROM DIFFERENT LEVEL (ELEVATION)',
       'PUSHING OR PULLING', 'FOREIGN MATTER (BODY) IN EYE(S)',
       'FALLING OR FLYING OBJECT',
       'FELLOW WORKER, PATIENT OR OTHER PERSON', 'STATIONARY OBJECT',
       'ON ICE OR SNOW', 'ABSORPTION, INGESTION OR INHALATION, NOC',
       'PERSON IN ACT OF A CRIME', 'ON STAIRS',
       'FROM LADDER OR SCAFFOLDING', 'SLIP, OR TRIP, DID NOT FALL',
       'JUMPING OR LEAPING', 'MOTOR VEHICLE', 'RUBBED OR ABRADED, NOC',
       'REACHING', 'OBJECT HANDLED', 'ELECTRICAL CURRENT',
       'HOLDING OR CARRYING', 'CAUGHT IN, UNDER OR BETWEEN, NOC',
       'FIRE OR FLAME', 'HOT OBJECTS OR SUBSTANCES',
       'POWERED HAND TOOL, APPLIANCE', 'STEAM OR HOT FLUIDS',
       'STRIKING AGAINST OR STEPPING ON, NOC', 'MACHINE OR MACHINERY',
       'COLD OBJECTS OR SUBSTANCES', 'BROKEN GLASS', 'CUMULATIVE, NOC',
       'COLLISION WITH A FIXED OBJECT', 'STEPPING ON SHARP OBJECT',
       'OBJECT HANDLED BY OTHERS', 'DUST, GASES, FUMES OR VAPORS',
       'CONTACT WITH, NOC', 'SANDING, SCRAPING, CLEANING OPERATION',
       'CONTINUAL NOISE', 'OTHER THAN PHYSICAL CAUSE OF INJURY',
       'USING TOOL OR MACHINERY', 'ANIMAL OR INSECT',
       'MOVING PARTS OF MACHINE', 'GUNSHOT', 'MOVING PART OF MACHINE',
       'TEMPERATURE EXTREMES', 'HAND TOOL OR MACHINE IN USE',
       'VEHICLE UPSET', 'INTO OPENINGS', 'WIELDING OR THROWING',
       'COLLAPSING MATERIALS (SLIDES OF EARTH)', 'TERRORISM', 'PANDEMIC',
       'WELDING OPERATION', 'NATURAL DISASTERS',
       'EXPLOSION OR FLARE BACK', 'RADIATION', 'CRASH OF RAIL VEHICLE',
       'MOLD', 'ABNORMAL AIR PRESSURE', 'CRASH OF WATER VEHICLE',
       'CRASH OF AIRPLANE') 
    
    nature_injury_descript = ('CONTUSION', 'SPRAIN OR TEAR', 'CONCUSSION', 'PUNCTURE',
       'LACERATION', 'ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC',
       'ALL OTHER SPECIFIC INJURIES, NOC', 'BURN', 'STRAIN OR TEAR',
       'FRACTURE', 'INFLAMMATION', 'FOREIGN BODY',
       'MULTIPLE PHYSICAL INJURIES ONLY', 'RUPTURE', 'DISLOCATION',
       'ALL OTHER CUMULATIVE INJURY, NOC', 'HERNIA', 'ANGINA PECTORIS',
       'CARPAL TUNNEL SYNDROME', 'NO PHYSICAL INJURY', 'INFECTION',
       'CRUSHING', 'SYNCOPE', 'POISONING - GENERAL (NOT OD OR CUMULATIVE',
       'RESPIRATORY DISORDERS', 'HEARING LOSS OR IMPAIRMENT',
       'MENTAL STRESS', 'SEVERANCE', 'ELECTRIC SHOCK', 'LOSS OF HEARING',
       'DUST DISEASE, NOC', 'DERMATITIS', 'ASPHYXIATION',
       'MENTAL DISORDER', 'CONTAGIOUS DISEASE', 'AMPUTATION',
       'MYOCARDIAL INFARCTION',
       'POISONING - CHEMICAL, (OTHER THAN METALS)',
       'MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL',
       'VISION LOSS', 'VASCULAR', 'COVID-19', 'CANCER',
       'HEAT PROSTRATION', 'AIDS', 'ASBESTOSIS', 'POISONING - METAL',
       'VDT - RELATED DISEASES', 'FREEZING', 'BLACK LUNG', 'SILICOSIS',
       'ADVERSE REACTION TO A VACCINATION OR INOCULATION', 'HEPATITIS C',
       'RADIATION', 'ENUCLEATION', 'BYSSINOSIS')
    
    part_of_body_descript = ('BUTTOCKS', 'SHOULDER(S)', 'MULTIPLE HEAD INJURY', 'FINGER(S)',
       'LUNGS', 'EYE(S)', 'ANKLE', 'THUMB', 'LOWER BACK AREA',
       'ABDOMEN INCLUDING GROIN', 'LOWER LEG', 'HIP', 'UPPER LEG',
       'MOUTH', 'KNEE', 'WRIST', 'SPINAL CORD', 'HAND', 'SOFT TISSUE',
       'UPPER ARM', 'ELBOW', 'MULTIPLE UPPER EXTREMITIES',
       'MULTIPLE BODY PARTS (INCLUDING BODY',
       'BODY SYSTEMS AND MULTIPLE BODY SYSTEMS', 'MULTIPLE NECK INJURY',
       'FOOT', 'EAR(S)', 'MULTIPLE LOWER EXTREMITIES', 'DISC',
       'LOWER ARM', 'MULTIPLE', 'CHEST', 'UPPER BACK AREA', 'SKULL',
       'WRIST (S) & HAND(S)', 'TOES', 'FACIAL BONES',
       'NO PHYSICAL INJURY', 'MULTIPLE TRUNK', 'WHOLE BODY',
       'INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED', 'PELVIS',
       'NOSE', 'GREAT TOE', 'INTERNAL ORGANS', 'VERTEBRAE',
       'LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA', 'BRAIN',
       'SACRUM AND COCCYX', 'ARTIFICIAL APPLIANCE', 'TEETH', 'LARYNX',
       'HEART', 'TRACHEA')
    
    indust_code_descript = ('RETAIL TRADE', 'CONSTRUCTION',
       'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT',
       'HEALTH CARE AND SOCIAL ASSISTANCE',
       'ACCOMMODATION AND FOOD SERVICES', 'EDUCATIONAL SERVICES',
       'MANUFACTURING', 'TRANSPORTATION AND WAREHOUSING',
       'WHOLESALE TRADE', 'REAL ESTATE AND RENTAL AND LEASING',
       'FINANCE AND INSURANCE',
       'OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)',
       'PUBLIC ADMINISTRATION', 'INFORMATION',
       'PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES',
       'ARTS, ENTERTAINMENT, AND RECREATION','UTILITIES', 'MINING',
       'AGRICULTURE, FORESTRY, FISHING AND HUNTING',
       'MANAGEMENT OF COMPANIES AND ENTERPRISES')
    
    carrier_name_values = ('NEW HAMPSHIRE INSURANCE CO', 'ZURICH AMERICAN INSURANCE CO',
       'INDEMNITY INSURANCE CO OF', 'MARATHON CENTRAL SCHOOL DIST',
       'CAMBRIDGE CENTRAL SCHOOL', 'HERMON-DEKALB CENTRAL')
    
    
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
            list(nature_injury_descript),
            help="Select the nature of injury.",
        )
        with col2:
            part_of_body = st.selectbox(
                "Part of Body Injured",
                part_of_body_descript,
                help="Select the injured body part.",
        )
            cause_injury = st.selectbox(
                "Cause of Injury",
                cause_injury_descript,
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
            indust_code_descript,
    )




