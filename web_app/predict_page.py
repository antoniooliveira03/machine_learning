import streamlit as st
import pickle
import gzip
import numpy as np
import datetime

def load_model():
    with gzip.open("trained_model.pkl.gz", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

#Initializing of the encodind methods
rf_loaded = data ["model"]
freq = data ["freq"]
le_injury_cause = data["le_injury_cause"]
le_injury_nature = data["le_injury_nature"]
le_body = data["le_body"]
le_industry = data ["le_industry"]
label_mapping = data["label_mapping"]

def show_predict_page():
    st.title("Grant or not Grant Compensation Benefit")
    st.write("""### We need some information to predict the claim injury type""")

    binary_y_n = ("Yes", "No")

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
    
    first_hearing_years = ('0', '2023', '2021', '2022', '2020', '2024')
    
    

    birth_year = st.slider("Worker Birth Year",1944, 2024, 1944)
    accident_year = st.slider("Accident Year",1966, 2021, 1966)
    accident_month = st.slider("Accident Month", 1, 12, 1)
    first_hearing = st.selectbox("First hearing Year", first_hearing_years)
    C_2_day = st.slider("In wich day did you receive the Employer's Report of Work-Related Injury/Illness",
                        1, 31, 1)
    C_3_date = st.selectbox("Did the Employee Claim Form was received?", binary_y_n)
    IME_4 = st.slider("What was the number of IMS-4 forms received per claim of the worker", 
                      0,30,1)
    attorney = st.selectbox("Is the claim being represented by an Attorney?", binary_y_n)
    avg_weekly_wage = st.number_input(label="Enter the average weekly wage of the coworker",
                                      min_value=0,
                                      max_value= 2828079,
                                      value=500,
                                      step=1)
    carrier_name = st.selectbox("Choose the primary insurance provider responsible for providing (Carrier Name)",
                                carrier_name_values)
    cause_injury = st.selectbox("What was the case of the injury?", cause_injury_descript)
    nature_injury = st.selectbox("Whats was the nature of injury?", nature_injury_descript)
    part_of_body = st.selectbox("What part of the body was the most injured?", part_of_body_descript)
    indust_code = st.selectbox("What industry is the claim comes from?", indust_code_descript)

    ok = st.button("Predict the Claim Injury Type")
    if ok:
         users_input = np.array([[birth_year, accident_year, accident_month, first_hearing, C_2_day, C_3_date, IME_4, attorney, avg_weekly_wage, carrier_name,  
                     cause_injury, nature_injury, part_of_body, indust_code]])   
         
         users_input[0, 5] = 0 if users_input[0, 5] == "No" else 1 
         users_input[0, 7] = 1 if users_input[0, 7] == 'Yes' else 0 
         users_input[0, 9] = freq.get(users_input[0, 9], 0)  
         users_input[0, 10] = le_injury_cause.transform([users_input[0, 10]])[0]  
         users_input[0, 11] = le_injury_nature.transform([users_input[0, 11]])[0]  
         users_input[0, 12] = le_body.transform([users_input[0, 12]])[0] 
         users_input[0, 13] = le_industry.transform([users_input[0, 13]])[0] 
         users_input = users_input.astype(float)

         # Automatically reshape the input to 2D to not get an error due to the sklearn library
         reshaped_input = users_input.reshape(1, -1)

         # Prediction
         pred = rf_loaded.predict(reshaped_input)

         # Reverse label mapping for prediction
         reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reversing the original mapping

         # Convert numeric prediction back to string
         predicted_label = reverse_label_mapping.get(pred[0], "Unknown")
         st.subheader(f"The Claim injury type is is {predicted_label}")

        

        




        
        
        
        
        
        
        
        
        
        
        
        
         
