standard_bandgap = {
    "Si": 1.12,
    "Ge": 0.66,
    "GaAs": 1.42,
    "AlN": 6.2
}
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("🔬 Inorganic Bandgap Prediction App")

# ===============================
# Element properties
# ===============================
elem_props = {
 'H': (1, 2.20, 53), 'Li': (3, 0.98, 167), 'Be': (4, 1.57, 112),
 'B': (5, 2.04, 87), 'C': (6, 2.55, 67), 'N': (7, 3.04, 56),
 'O': (8, 3.44, 48), 'F': (9, 3.98, 42), 'Na': (11, 0.93, 190),
 'Mg': (12, 1.31, 145), 'Al': (13, 1.61, 118), 'Si': (14, 1.90, 111)
}

# ===============================
# Formula parser
# ===============================
def parse_formula(formula):
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    comp = Counter()
    for el, cnt in tokens:
        cnt = int(cnt) if cnt != '' else 1
        comp[el] += cnt
    return dict(comp)

# ===============================
# Feature extractor
# ===============================
def featurize_formula(formula):
    comp = parse_formula(formula)
    total_atoms = sum(comp.values())

    zn_mean = 0
    en_mean = 0

    for el, cnt in comp.items():
        z, en, _ = elem_props.get(el, (0, 1.5, 150))
        weight = cnt / total_atoms
        zn_mean += weight * z
        en_mean += weight * en

    return [zn_mean, en_mean]

# ===============================
# Upload dataset
# ===============================
uploaded_file = st.file_uploader("Upload CSV with 'formula' and 'band_gap' columns")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # ===============================
    # Feature engineering
    # ===============================
    st.subheader("Feature Engineering")

    X = np.array([featurize_formula(f) for f in df['formula']])
    y = df['band_gap'].values

    st.write("Features extracted successfully")

    # ===============================
    # Train model
    # ===============================
    if st.button("Train Model"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        st.subheader("Model Performance")
        st.write(f"MAE: {mae:.3f}")
        st.write(f"RMSE: {rmse:.3f}")
        st.write(f"R2 Score: {r2:.3f}")
        st.subheader("Model Performance")
        st.write(f"MAE: {mae:.3f}")
        st.write(f"RMSE: {rmse:.3f}")
        st.write(f"R2 Score: {r2:.3f}")

# 🔥 ADD THIS BLOCK HERE
     if r2 > 0.9:
    st.success("Model is highly reliable (Excellent fit)")
    elif r2 > 0.75:
    st.info("Model has good predictive power")
else:
    st.warning("Model performance needs improvement")

        # ===============================
        # Prediction section
        # ===============================
        st.subheader("Predict New Compound")

        formula_input = st.text_input("Enter formula (e.g., SiO2)")

        if st.button("Predict"):
    feat = np.array(featurize_formula(formula_input)).reshape(1, -1)
    prediction = model.predict(feat)[0]

    st.subheader("Prediction Results")

    # -----------------------------
    # Predicted value
    # -----------------------------
    st.write(f"Predicted Band Gap: {prediction:.3f} eV")

    # -----------------------------
    # Extract main material (for lookup)
    # -----------------------------
    material_key = re.findall(r'[A-Z][a-z]?', formula_input)
    material_key = material_key[0] if material_key else None

    # -----------------------------
    # Standard value + comparison
    # -----------------------------
    if material_key in standard_bandgap:
        actual = standard_bandgap[material_key]
        error = abs(actual - prediction)

        st.write(f"Standard Band Gap: {actual} eV")
        st.write(f"Error: {error:.3f}")

        # -----------------------------
        # Conclusion
        # -----------------------------
        if error < 0.1:
            conclusion = "Excellent prediction"
            st.success(conclusion)
        elif error < 0.3:
            conclusion = "Good prediction"
            st.info(conclusion)
        else:
            conclusion = "Needs improvement"
            st.warning(conclusion)

        # -----------------------------
        # FINAL TABLE (VERY IMPORTANT ⭐)
        # -----------------------------
        result_df = pd.DataFrame({
            "Material": [material_key],
            "Standard Bandgap (eV)": [actual],
            "Predicted Bandgap (eV)": [prediction],
            "Error": [error],
            "Conclusion": [conclusion]
        })

        st.subheader("Final Comparison Table")
        st.dataframe(result_df)

    else:
        st.warning("Standard bandgap not available for this material")
