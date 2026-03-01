# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt

# # -------------------------------
# # Load model & scaler
# # -------------------------------
# with open("lightgbm_claim_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # -------------------------------
# # Page config
# # -------------------------------
# st.set_page_config(
#     page_title="Car Insurance Claim Prediction",
#     layout="wide"
# )

# # -------------------------------
# # Sidebar navigation
# # -------------------------------
# st.sidebar.title("Navigation")
# page = st.sidebar.radio(
#     "Go to",
#     ["Project Overview", "EDA Insights", "Claim Prediction"]
# )

# # ===============================
# # PAGE 1: PROJECT OVERVIEW
# # ===============================
# if page == "Project Overview":
#     st.title("🚗 Car Insurance Claim Prediction")

#     st.markdown("""
#     ### Problem Statement
#     Predict whether a customer will make an insurance claim in the next policy period
#     using demographic, vehicle, and policy-related features.
#     """)

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Dataset Size", "58,592 rows")
#     col2.metric("Target Variable", "is_claim (0 / 1)")
#     col3.metric("Final Model", "LightGBM")

#     st.markdown("""
#     ### Business Use Cases
#     - Fraud Prevention  
#     - Premium Pricing Optimization  
#     - Customer Risk Segmentation  
#     - Claims Resource Planning  
#     """)

# # ===============================
# # PAGE 2: EDA INSIGHTS
# # ===============================
# elif page == "EDA Insights":
#     st.title("📊 Exploratory Data Analysis")

#     # Load small sample or full data
#     data = pd.read_csv("data_train.csv")

#     col1, col2 = st.columns(2)

#     with col1:
#        st.subheader("Claim vs No Claim Distribution")
#     fig, ax = plt.subplots()
#     data['is_claim'].value_counts().plot(kind='bar', ax=ax)
#     ax.set_xlabel("is_claim")
#     ax.set_ylabel("Count")
#     st.pyplot(fig)

#     with col2:
#         st.subheader("Age of Car vs Claim")

#     fig, ax = plt.subplots()
#     data.boxplot(column='age_of_car', by='is_claim', ax=ax)
#     plt.suptitle("")
#     ax.set_title("Age of Car vs Claim")
#     st.pyplot(fig)
#     st.subheader("Claim Rate by Vehicle Segment")

#     segment_claim = data.groupby('segment')['is_claim'].mean()

#     fig, ax = plt.subplots()
#     segment_claim.plot(kind='bar', ax=ax)
#     ax.set_ylabel("Claim Rate")
#     st.pyplot(fig)
# # ===============================
# # PAGE 3: CLAIM PREDICTION
# # ===============================
# else:
#     st.title("🔮 Insurance Claim Prediction")

#     st.markdown("### Enter Customer & Vehicle Details")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         policy_tenure = st.number_input("Policy Tenure", 0.0, 2.0, 0.5)
#         age_of_car = st.number_input("Age of Car", 0.0, 1.0, 0.05)
#         population_density = st.number_input("Population Density", 100, 80000, 10000)

#     with col2:
#         segment = st.selectbox("Vehicle Segment", [0, 1, 2, 3, 4, 5])
#         fuel_type = st.selectbox("Fuel Type", [1, 2, 3])
#         airbags = st.selectbox("Airbags", [1, 2, 6])

#     with col3:
#         is_esc = st.selectbox("ESC Available", [0, 1])
#         is_tpms = st.selectbox("TPMS Available", [0, 1])
#         is_brake_assist = st.selectbox("Brake Assist", [0, 1])

#     if st.button("Predict Claim Risk"):
#         input_data = pd.DataFrame([[
#             policy_tenure,
#             age_of_car,
#             population_density,
#             segment,
#             fuel_type,
#             airbags,
#             is_esc,
#             is_tpms,
#             is_brake_assist
#         ]], columns=[
#             'policy_tenure',
#             'age_of_car',
#             'population_density',
#             'segment',
#             'fuel_type',
#             'airbags',
#             'is_esc',
#             'is_tpms',
#             'is_brake_assist'
#         ])

#         # scale numeric columns
#         num_cols = ['policy_tenure', 'age_of_car', 'population_density']
#         input_data[num_cols] = scaler.transform(input_data[num_cols])

#         prob = model.predict_proba(input_data)[0][1]

#         st.subheader("Prediction Result")
#         st.write(f"### Claim Probability: **{prob:.2%}**")

#         if prob < 0.3:
#             st.success("🟢 Low Risk Customer")
#         elif prob < 0.6:
#             st.warning("🟡 Medium Risk Customer")
#         else:
#             st.error("🔴 High Risk Customer")




import streamlit as st
import pandas as pd
import pickle

# =========================
# Load Model & Scaler
# =========================
model = pickle.load(open("lightgbm_claim_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Car Insurance Prediction", layout="wide")

st.title("🚗 Car Insurance Prediction Dashboard")
st.write("Predict whether a customer is likely to purchase car insurance")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs([
    "🔮 Prediction",
    "📊 Project Details",
    "💻 About This Code"
])

# =========================
# TAB 1 — Prediction
# =========================
with tab1:
    st.header("Enter Customer Details")

    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     age = st.number_input(
    #         "Age",
    #         min_value=18,
    #         max_value=80,
    #         value=30,
    #         step=1
    #     )

    #     gender = st.selectbox("Gender", ["Male", "Female"])
    #     gender = 1 if gender == "Male" else 0

    #     driving_license = st.selectbox("Driving License", ["Yes", "No"])
    #     driving_license = 1 if driving_license == "Yes" else 0

    # with col2:
    #     region_code = st.number_input(
    #         "Region Code",
    #         min_value=1,
    #         value=28,
    #         step=1
    #     )

    #     previously_insured = st.selectbox(
    #         "Previously Insured",
    #         ["Yes", "No"]
    #     )
    #     previously_insured = 1 if previously_insured == "Yes" else 0

    #     vehicle_age = st.selectbox(
    #         "Vehicle Age",
    #         ["< 1 Year", "1-2 Years", "> 2 Years"]
    #     )

    #     vehicle_age_map = {
    #         "< 1 Year": 0,
    #         "1-2 Years": 1,
    #         "> 2 Years": 2
    #     }
    #     vehicle_age = vehicle_age_map[vehicle_age]

    # with col3:
    #     vehicle_damage = st.selectbox(
    #         "Vehicle Damage",
    #         ["Yes", "No"]
    #     )
    #     vehicle_damage = 1 if vehicle_damage == "Yes" else 0

    #     annual_premium = st.number_input(
    #         "Annual Premium (₹)",
    #         min_value=2000,
    #         value=30000,
    #         step=1000
    #     )

    #     policy_sales_channel = st.number_input(
    #         "Policy Sales Channel",
    #         min_value=1,
    #         value=152,
    #         step=1
    #     )

    #     vintage = st.number_input(
    #         "Customer Vintage (Days)",
    #         min_value=0,
    #         value=150,
    #         step=1
    #     )

    # # =========================
    # # Predict Button
    # # =========================
    # if st.button("Predict Insurance Purchase"):

    #     input_df = pd.DataFrame([[  
    #         age,
    #         gender,
    #         driving_license,
    #         region_code,
    #         previously_insured,
    #         vehicle_age,
    #         vehicle_damage,
    #         annual_premium,
    #         policy_sales_channel,
    #         vintage
    #     ]],
    #     columns=[
    #         'Age',
    #         'Gender',
    #         'Driving_License',
    #         'Region_Code',
    #         'Previously_Insured',
    #         'Vehicle_Age',
    #         'Vehicle_Damage',
    #         'Annual_Premium',
    #         'Policy_Sales_Channel',
    #         'Vintage'
    #     ])

    #     # =========================
    #     # Scaling
    #     # =========================
    #     num_cols = ['Age', 'Annual_Premium', 'Vintage']
    #     input_df[num_cols] = scaler.transform(input_df[num_cols])

    #     # =========================
    #     # Prediction
    #     # =========================
    #     prediction = model.predict(input_df)[0]

    #     if prediction == 1:
    #         st.success("✅ Customer is likely to purchase Car Insurance")
    #     else:
    #         st.error("❌ Customer is not likely to purchase Car Insurance")
    # =========================
    # USER INPUTS
    # =========================

    yes_no_map = {"No": 0, "Yes": 1}

    policy_tenure = st.number_input("Policy Tenure (Years)", 0.0, 10.0, 1.0)
    age_of_car = st.number_input("Age of Car (Years)", 0.0, 20.0, 3.0)
    age_of_policyholder = st.number_input("Policyholder Age", 18, 90, 35)

    area_cluster = st.selectbox("Area Cluster", list(range(1, 7)))
    segment = st.selectbox("Car Segment", list(range(1, 6)))
    fuel_type = st.selectbox("Fuel Type", [1, 2, 3])  # 1=CNG,2=Petrol,3=Diesel
    transmission_type = st.selectbox("Transmission Type", [0, 1])  # 0=Manual,1=Auto

    population_density = st.number_input("Population Density", 100, 50000, 1000)

    max_power = st.number_input("Max Power", 20.0, 500.0, 80.0)
    max_torque = st.number_input("Max Torque", 50.0, 600.0, 120.0)

    displacement = st.number_input("Engine Displacement", 500, 5000, 1200)
    gross_weight = st.number_input("Gross Weight", 500, 4000, 1500)

    airbags = st.number_input("Airbags", 0, 10, 2)
    ncap_rating = st.selectbox("NCAP Rating", [0, 1, 2, 3, 4, 5])

    # Binary features
    is_esc = yes_no_map[st.selectbox("ESC", ["No", "Yes"])]
    is_adjustable_steering = yes_no_map[st.selectbox("Adjustable Steering", ["No", "Yes"])]
    is_tpms = yes_no_map[st.selectbox("TPMS", ["No", "Yes"])]
    is_parking_sensors = yes_no_map[st.selectbox("Parking Sensors", ["No", "Yes"])]
    is_parking_camera = yes_no_map[st.selectbox("Parking Camera", ["No", "Yes"])]
    is_front_fog_lights = yes_no_map[st.selectbox("Front Fog Lights", ["No", "Yes"])]
    is_rear_window_wiper = yes_no_map[st.selectbox("Rear Window Wiper", ["No", "Yes"])]
    is_rear_window_washer = yes_no_map[st.selectbox("Rear Window Washer", ["No", "Yes"])]
    is_rear_window_defogger = yes_no_map[st.selectbox("Rear Window Defogger", ["No", "Yes"])]
    is_brake_assist = yes_no_map[st.selectbox("Brake Assist", ["No", "Yes"])]
    is_ecw = yes_no_map[st.selectbox("ECW", ["No", "Yes"])]
    is_speed_alert = yes_no_map[st.selectbox("Speed Alert", ["No", "Yes"])]

    # =========================
    # FEATURE ORDER (VERY IMPORTANT)
    # =========================
    feature_order = [
        'policy_tenure',
        'age_of_car',
        'age_of_policyholder',
        'area_cluster',
        'population_density',
        'segment',
        'fuel_type',
        'max_torque',
        'max_power',
        'airbags',
        'is_esc',
        'is_adjustable_steering',
        'is_tpms',
        'is_parking_sensors',
        'is_parking_camera',
        'displacement',
        'transmission_type',
        'gross_weight',
        'is_front_fog_lights',
        'is_rear_window_wiper',
        'is_rear_window_washer',
        'is_rear_window_defogger',
        'is_brake_assist',
        'is_ecw',
        'is_speed_alert',
        'ncap_rating'
    ]

    # =========================
    # CREATE INPUT DATAFRAME
    # =========================
    input_df = pd.DataFrame([[  
        policy_tenure,
        age_of_car,
        age_of_policyholder,
        area_cluster,
        population_density,
        segment,
        fuel_type,
        max_torque,
        max_power,
        airbags,
        is_esc,
        is_adjustable_steering,
        is_tpms,
        is_parking_sensors,
        is_parking_camera,
        displacement,
        transmission_type,
        gross_weight,
        is_front_fog_lights,
        is_rear_window_wiper,
        is_rear_window_washer,
        is_rear_window_defogger,
        is_brake_assist,
        is_ecw,
        is_speed_alert,
        ncap_rating
    ]], columns=feature_order)

    # =========================
    # SCALE NUMERIC COLUMNS
    # =========================
    num_cols = [
        'policy_tenure',
        'age_of_car',
        'age_of_policyholder',
        'population_density',
        'displacement',
        'gross_weight',
        'max_power',
        'max_torque'
    ]

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict Insurance Claim"):

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("🚨 Claim Likely")
        else:
            st.success("✅ No Claim Expected")
# =========================
# TAB 2 — Project Details
# =========================
with tab2:
    st.header("📊 Project Overview")

    st.subheader("🧩 Problem Statement")
    st.info("""
    The objective of this project is to predict whether a customer
    will purchase car insurance based on demographic,
    vehicle, and policy-related features.
    """)

    st.divider()

    st.subheader("🗂️ Dataset Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Features", "10")
    col2.metric("Records", "3L+")
    col3.metric("Type", "Insurance Dataset")

    st.divider()

    st.subheader("🧹 Data Preprocessing")
    with st.expander("View Steps"):
        st.success("""
        • Handled missing values  
        • Encoded categorical variables  
        • Feature scaling applied  
        • Outliers treated  
        """)

    st.divider()

    st.subheader("🤖 Models Used")
    st.write("""
    • Logistic Regression  
    • Random Forest Classifier  
    • XGBoost Classifier  
    """)

    st.divider()

    st.subheader("🏆 Final Model Selected")
    st.success("""
    Random Forest Classifier selected based on:
    • High ROC-AUC  
    • Balanced Precision & Recall  
    • Stable performance  
    """)

# =========================
# TAB 3 — About This Code
# =========================
with tab3:
    st.header("💻 About This Dashboard")

    st.subheader("🛠️ Technologies Used")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        • Python  
        • Pandas  
        • NumPy  
        """)
    with col2:
        st.info("""
        • Scikit-learn  
        • Streamlit  
        • Pickle  
        """)

    st.divider()

    st.subheader("⚙️ Workflow")
    st.success("""
    1️⃣ Data Collection  
    2️⃣ Data Cleaning  
    3️⃣ Feature Engineering  
    4️⃣ Model Training  
    5️⃣ Model Evaluation  
    6️⃣ Model Saving  
    7️⃣ Dashboard Deployment  
    """)

    st.divider()

    st.subheader("🎯 Purpose")
    st.warning("""
    • Predict customer interest in car insurance  
    • Assist insurance companies in targeting customers  
    • Improve conversion rates  
    """)