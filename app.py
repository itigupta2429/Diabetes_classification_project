"""
app.py — Diabetes 30-Day Readmission Predictor
------------------------------------------------
Streamlit web app that predicts whether a diabetic hospital patient
will be readmitted within 30 days of discharge.

Dataset: UCI 130-US Hospitals Diabetes dataset (101,766 encounters)
Model  : Random Forest with balanced class weights
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
)

# ── Load model & metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipe = joblib.load("model.pkl")
    with open("feature_meta.json") as f:
        meta = json.load(f)
    return pipe, meta

pipe, meta = load_model()
numeric_cols     = meta["numeric_cols"]
categorical_cols = meta["categorical_cols"]

# ── Sidebar — About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=80)
    st.title("About This App")
    st.markdown("""
    This tool predicts whether a **diabetic patient** will be readmitted to hospital
    within **30 days** of discharge.

    **Model:** Random Forest (class-balanced)
    **Dataset:** 130 US hospitals · 101,766 encounters
    **ROC-AUC:** ~0.65

    ---
    **Feature categories**
    - 🧑 Demographics (age, race, gender)
    - 🏥 Hospital encounter details
    - 💊 Medications & drug changes
    - 🩺 Diagnoses & lab results
    - 📋 Patient history

    ---
    ⚠️ *For research/demo purposes only. Not a clinical decision tool.*
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 Diabetes 30-Day Readmission Predictor")
st.markdown(
    "Fill in the patient's details below and click **Predict** to estimate "
    "the probability of hospital readmission within 30 days."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("patient_form"):

    # ── Section 1: Demographics ───────────────────────────────────────────────
    st.subheader("👤 Demographics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Age group", [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
        ], index=6)

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"])

    with col3:
        race = st.selectbox("Race", [
            "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
        ])

    st.divider()

    # ── Section 2: Hospital Encounter ─────────────────────────────────────────
    st.subheader("🏥 Hospital Encounter")
    col4, col5, col6 = st.columns(3)

    with col4:
        time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
        num_lab_procedures = st.slider("Number of lab procedures", 1, 132, 45)
        num_procedures = st.slider("Number of procedures", 0, 6, 1)

    with col5:
        num_medications = st.slider("Number of medications", 1, 81, 16)
        number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
        admission_type_id = st.selectbox(
            "Admission type ID",
            [1, 2, 3, 4, 5, 6, 7, 8],
            index=0,
            help="1=Emergency, 2=Urgent, 3=Elective, 4=Newborn, 5=Not Available, 6=NULL, 7=Trauma, 8=Not Mapped"
        )

    with col6:
        discharge_disposition_id = st.number_input(
            "Discharge disposition ID", min_value=1, max_value=29, value=1,
            help="1=Discharged to home, 3=SNF, 6=Home w/ health service, etc."
        )
        admission_source_id = st.number_input(
            "Admission source ID", min_value=1, max_value=25, value=7,
            help="7=Emergency Room, 1=Physician Referral, 4=Transfer, etc."
        )
        weight = st.number_input(
            "Weight category (coded)", min_value=0.0, max_value=200.0, value=75.0,
            help="Numeric weight category from the dataset encoding"
        )

    st.divider()

    # ── Section 3: Prior Encounters & History ─────────────────────────────────
    st.subheader("📋 Patient History")
    col7, col8 = st.columns(2)

    with col7:
        number_outpatient = st.slider("Outpatient visits (past year)", 0, 42, 0)
        number_emergency  = st.slider("Emergency visits (past year)", 0, 76, 0)
        number_inpatient  = st.slider("Inpatient visits (past year)", 0, 21, 0)

    with col8:
        patient_prior_encounters = st.slider("Prior encounters in dataset", 0, 50, 1)
        patient_historical_readmission_rate = st.slider(
            "Historical readmission rate (0–1)", 0.0, 1.0, 0.1, step=0.01
        )

    st.divider()

    # ── Section 4: Lab Results ────────────────────────────────────────────────
    st.subheader("🩺 Lab Results")
    col9, col10 = st.columns(2)

    with col9:
        max_glu_serum = st.selectbox(
            "Max glucose serum result",
            ["NotPerformed", "Norm", ">200", ">300"]
        )
    with col10:
        A1Cresult = st.selectbox(
            "HbA1c (A1C) result",
            ["NotPerformed", "Norm", ">7", ">8"]
        )

    st.divider()

    # ── Section 5: Diagnoses ──────────────────────────────────────────────────
    st.subheader("🏷️ Diagnoses")
    DIAG_CATEGORIES = [
        "Diseases Of The Circulatory System",
        "Diseases Of The Respiratory System",
        "Diseases Of The Digestive System",
        "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders",
        "Diseases Of The Musculoskeletal System And Connective Tissue",
        "Diseases Of The Genitourinary System",
        "Diseases Of The Nervous System And Sense Organs",
        "Injury And Poisoning",
        "Neoplasms",
        "Infectious And Parasitic Diseases",
        "Mental Disorders",
        "Diseases Of The Blood And Blood-Forming Organs",
        "Diseases Of The Skin And Subcutaneous Tissue",
        "Complications Of Pregnancy, Childbirth, And The Puerperium",
        "Congenital Anomalies",
        "Symptoms, Signs, And Ill-Defined Conditions",
        "Supplementary Classification Of External Causes Of Injury And Poisoning",
        "Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services",
        "Unknown",
    ]
    col11, col12, col13 = st.columns(3)
    with col11:
        diag_1_category = st.selectbox("Primary diagnosis", DIAG_CATEGORIES, index=0)
    with col12:
        diag_2_category = st.selectbox("Secondary diagnosis", DIAG_CATEGORIES, index=0)
    with col13:
        diag_3_category = st.selectbox("Tertiary diagnosis", DIAG_CATEGORIES, index=0)

    st.divider()

    # ── Section 6: Medical Specialty ─────────────────────────────────────────
    st.subheader("🏨 Medical Specialty")
    medical_specialty = st.selectbox("Attending medical specialty", [
        "InternalMedicine", "Family/GeneralPractice", "Cardiology",
        "Surgery-General", "Orthopedics", "Gastroenterology",
        "Emergency/Trauma", "Nephrology", "Pulmonology",
        "Endocrinology", "Neurology", "Oncology",
        "ObstetricsandGynecology", "Hematology/Oncology",
        "Psychiatry", "Hospitalist", "Radiology", "Unknown",
        "AllergyandImmunology", "Anesthesiology", "Dermatology",
        "InfectiousDiseases", "Urology", "Rheumatology",
        "PhysicalMedicineandRehabilitation", "Other",
    ])
    # Map "Other" to a safe fallback for the encoder
    if medical_specialty == "Other":
        medical_specialty = "Unknown"

    st.divider()

    # ── Section 7: Medications ────────────────────────────────────────────────
    st.subheader("💊 Medications")
    MED_OPTIONS = ["No", "Steady", "Up", "Down"]

    col14, col15, col16 = st.columns(3)
    with col14:
        metformin    = st.selectbox("Metformin",    MED_OPTIONS, index=1)
        insulin      = st.selectbox("Insulin",      MED_OPTIONS, index=1)
        glipizide    = st.selectbox("Glipizide",    MED_OPTIONS, index=0)
        glyburide    = st.selectbox("Glyburide",    MED_OPTIONS, index=0)
        glimepiride  = st.selectbox("Glimepiride",  MED_OPTIONS, index=0)

    with col15:
        pioglitazone  = st.selectbox("Pioglitazone",  MED_OPTIONS, index=0)
        rosiglitazone = st.selectbox("Rosiglitazone", MED_OPTIONS, index=0)
        repaglinide   = st.selectbox("Repaglinide",   MED_OPTIONS, index=0)
        nateglinide   = st.selectbox("Nateglinide",   MED_OPTIONS, index=0)
        acarbose      = st.selectbox("Acarbose",      MED_OPTIONS, index=0)

    with col16:
        miglitol          = st.selectbox("Miglitol",          MED_OPTIONS, index=0)
        chlorpropamide    = st.selectbox("Chlorpropamide",    MED_OPTIONS, index=0)
        acetohexamide     = st.selectbox("Acetohexamide",     ["No", "Steady"], index=0)
        tolbutamide       = st.selectbox("Tolbutamide",       ["No", "Steady"], index=0)
        troglitazone      = st.selectbox("Troglitazone",      ["No", "Steady"], index=0)
        tolazamide        = st.selectbox("Tolazamide",        ["No", "Steady", "Up"], index=0)
        glyburide_metformin    = st.selectbox("Glyburide-Metformin",    MED_OPTIONS, index=0)
        glipizide_metformin    = st.selectbox("Glipizide-Metformin",    ["No", "Steady"], index=0)
        glimepiride_pioglitazone = st.selectbox("Glimepiride-Pioglitazone", ["No", "Steady"], index=0)
        metformin_rosiglitazone  = st.selectbox("Metformin-Rosiglitazone",  ["No", "Steady"], index=0)
        metformin_pioglitazone   = st.selectbox("Metformin-Pioglitazone",   ["No", "Steady"], index=0)

    st.divider()

    # ── Section 8: Medication Change & Diabetes Med ──────────────────────────
    st.subheader("🔄 Medication Management")
    col17, col18 = st.columns(2)
    with col17:
        change      = st.selectbox("Medication change during encounter", ["No", "Ch"], index=0,
                                   help="Ch = change was made")
    with col18:
        diabetesMed = st.selectbox("Diabetes medication prescribed?", ["Yes", "No"], index=0)

    st.divider()

    # ── Submit ────────────────────────────────────────────────────────────────
    submitted = st.form_submit_button("🔍 Predict Readmission Risk", use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────────────────
if submitted:
    # Build input row in the EXACT column order the model was trained on
    input_data = {
        # Numeric
        "weight":                              weight,
        "admission_type_id":                   int(admission_type_id),
        "discharge_disposition_id":            int(discharge_disposition_id),
        "admission_source_id":                 int(admission_source_id),
        "time_in_hospital":                    time_in_hospital,
        "num_lab_procedures":                  num_lab_procedures,
        "num_procedures":                      num_procedures,
        "num_medications":                     num_medications,
        "number_outpatient":                   number_outpatient,
        "number_emergency":                    number_emergency,
        "number_inpatient":                    number_inpatient,
        "number_diagnoses":                    number_diagnoses,
        "patient_prior_encounters":            patient_prior_encounters,
        "patient_historical_readmission_rate": patient_historical_readmission_rate,
        # Categorical
        "race":                         race,
        "gender":                       gender,
        "age":                          age,
        "medical_specialty":            medical_specialty,
        "max_glu_serum":                max_glu_serum,
        "A1Cresult":                    A1Cresult,
        "metformin":                    metformin,
        "repaglinide":                  repaglinide,
        "nateglinide":                  nateglinide,
        "chlorpropamide":               chlorpropamide,
        "glimepiride":                  glimepiride,
        "acetohexamide":                acetohexamide,
        "glipizide":                    glipizide,
        "glyburide":                    glyburide,
        "tolbutamide":                  tolbutamide,
        "pioglitazone":                 pioglitazone,
        "rosiglitazone":                rosiglitazone,
        "acarbose":                     acarbose,
        "miglitol":                     miglitol,
        "troglitazone":                 troglitazone,
        "tolazamide":                   tolazamide,
        "insulin":                      insulin,
        "glyburide-metformin":          glyburide_metformin,
        "glipizide-metformin":          glipizide_metformin,
        "glimepiride-pioglitazone":     glimepiride_pioglitazone,
        "metformin-rosiglitazone":      metformin_rosiglitazone,
        "metformin-pioglitazone":       metformin_pioglitazone,
        "change":                       change,
        "diabetesMed":                  diabetesMed,
        "diag_1_category":              diag_1_category,
        "diag_2_category":              diag_2_category,
        "diag_3_category":              diag_3_category,
    }

    input_df = pd.DataFrame([input_data])

    # Predict
    prob = pipe.predict_proba(input_df)[0][1]
    pred = pipe.predict(input_df)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        if prob >= 0.5:
            st.error(f"⚠️ **HIGH RISK**\n\nReadmission within 30 days is **likely**.")
        elif prob >= 0.25:
            st.warning(f"🟡 **MODERATE RISK**\n\nReadmission within 30 days is **possible**.")
        else:
            st.success(f"✅ **LOW RISK**\n\nReadmission within 30 days is **unlikely**.")

    with col_res2:
        st.metric(
            label="Predicted Readmission Probability",
            value=f"{prob:.1%}",
        )
        st.progress(float(prob))

        st.markdown(f"""
        | | |
        |---|---|
        | **Model prediction** | {"Readmitted < 30 days" if pred == 1 else "Not readmitted < 30 days"} |
        | **Confidence** | {prob:.1%} chance of early readmission |
        | **Model** | Random Forest (class-balanced) |
        | **Training data** | 81,613 patient encounters |
        """)

    # Key risk factors note
    st.info(
        "💡 **Factors most associated with early readmission in this dataset:** "
        "number of inpatient visits, number of diagnoses, time in hospital, "
        "emergency visits, and oncology-related diagnoses."
    )

    st.caption("⚠️ This prediction is for research & demonstration purposes only. "
               "Always rely on clinical judgement for medical decisions.")
