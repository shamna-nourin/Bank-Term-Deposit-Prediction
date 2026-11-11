
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. LOAD MODEL AND ARTIFACTS
# ==========================================

@st.cache_resource
def load_artifacts():

    with open("banking_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return model, label_encoders, feature_columns


try:
    model, label_encoders, feature_columns = load_artifacts()

except FileNotFoundError as e:
    st.error(f"Required model file not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background-color: #f4f6f9;
}

/* Main Container */

.main .block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Main Title */

.title {
    font-size: 38px;
    font-weight: 800;
    color: #24364b !important;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */

.subtitle {
    font-size: 17px;
    color: #64748b !important;
    text-align: center;
    margin-bottom: 30px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #252631;
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* Sidebar Header */

[data-testid="stSidebar"] h2 {
    color: #ffffff !important;
}

/* Main Headings */

.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3 {
    color: #24364b !important;
}

/* Normal Text */

.stMarkdown p {
    color: #475569;
}

/* Prediction Button */

div.stButton > button {

    width: 100%;
    background-color: #24364b;
    color: white !important;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;

}

/* Button Hover */

div.stButton > button:hover {

    background-color: #fcb431;
    color: #24364b !important;
    border: none;

}

/* Prediction Card */

.prediction-card {

    background-color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 3px 15px rgba(0,0,0,0.08);
    margin-top: 20px;

}

/* Footer */

.footer {

    text-align: center;
    color: #64748b !important;
    font-size: 14px;
    margin-top: 50px;
    padding: 15px;

}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. HEADER
# ==========================================

st.markdown(
    '<div class="title">💼 Bank Term Deposit Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze client information and predict term deposit subscription'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 5. SIDEBAR INPUTS
# ==========================================

st.sidebar.header("📋 Client Information")

st.sidebar.write(
    "Enter the client details to predict subscription."
)


def user_input():

    input_dict = {}

    # Numerical Features

    input_dict['age'] = st.sidebar.slider(
        'Age',
        min_value=18,
        max_value=100,
        value=30
    )

    input_dict['balance'] = st.sidebar.number_input(
        'Balance',
        value=1000
    )

    input_dict['day'] = st.sidebar.slider(
        'Day Contacted',
        min_value=1,
        max_value=31,
        value=15
    )

    input_dict['duration'] = st.sidebar.number_input(
        'Call Duration (sec)',
        min_value=0,
        value=150
    )

    input_dict['campaign'] = st.sidebar.number_input(
        'Campaign Contacts',
        min_value=1,
        value=1
    )

    input_dict['pdays'] = st.sidebar.number_input(
        'Days Since Last Contact',
        min_value=-1,
        value=999
    )

    input_dict['previous'] = st.sidebar.number_input(
        'Previous Contacts',
        min_value=0,
        value=0
    )

    # Categorical Features

    for col, encoder in label_encoders.items():

        if col == 'target_y':
            continue

        selected = st.sidebar.selectbox(
            col.replace('_', ' ').capitalize(),
            options=encoder.classes_.tolist()
        )

        input_dict[col] = encoder.transform(
            [selected]
        )[0]

    return pd.DataFrame([input_dict])


# ==========================================
# 6. GET USER INPUT
# ==========================================

input_df = user_input()


# ==========================================
# 7. FEATURE ALIGNMENT
# ==========================================

input_df = input_df.reindex(
    columns=feature_columns
)

if input_df.isnull().any().any():

    missing_columns = input_df.columns[
        input_df.isnull().any()
    ].tolist()

    st.error(
        f"Missing input features: {missing_columns}"
    )

    st.stop()


# ==========================================
# 8. MAIN CONTENT
# ==========================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    st.markdown(
        "### 🔮 Prediction",
    )

    st.write(
        "Click below to predict whether the client "
        "will subscribe to a term deposit."
    )

    predict_button = st.button(
        "🔮 Predict Term Deposit",
        use_container_width=True
    )


# ==========================================
# 9. MODEL PREDICTION
# ==========================================

if predict_button:

    try:

        # Model trained using unscaled features.
        # Therefore, scaler is not applied here.

        prediction = model.predict(input_df)

        # Decode prediction

        target_encoder = label_encoders['target_y']

        prediction_label = target_encoder.inverse_transform(
            np.asarray(prediction, dtype=int)
        )[0]

        st.divider()

        # YES Prediction

        if prediction_label == 'yes':

            st.success(
                "✅ The client is likely to subscribe "
                "to a term deposit!"
            )

            st.balloons()

        # NO Prediction

        else:

            st.error(
                "❌ The client is unlikely to subscribe "
                "to a term deposit."
            )

        # Prediction Probability

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            st.markdown("### 📊 Prediction Probability")

            for class_value, probability in zip(
                model.classes_,
                probabilities
            ):

                original_label = target_encoder.inverse_transform(
                    [int(class_value)]
                )[0]

                percentage = probability * 100

                st.write(
                    f"**{original_label.upper()}** : "
                    f"{percentage:.2f}%"
                )

                st.progress(float(probability))

        # Display Input Data

        with st.expander("View Client Information"):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit
    <br>
    Bank Term Deposit Prediction System
</div>
""", unsafe_allow_html=True)
