# ============================================================
# AI-POWERED HEALTH ASSISTANT
# Streamlit Application
# ============================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ============================================================
# EVIDENCEMD API CONFIGURATION
# ============================================================

EVIDENCEMD_API_KEY ="----------"
EVIDENCEMD_API_URL = "https://evidencemd.ai/api/v1/chat/completions"
EVIDENCEMD_MODEL = "evidencemd-fast"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "health_model.joblib"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HEADER
# ============================================================

st.html("""
<div style="
    background: linear-gradient(135deg, #0f4c81, #1976d2);
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
">

    <div style="
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    ">
        🩺 AI-Powered Health Assistant
    </div>

    <div style="
        font-size: 17px;
        line-height: 1.6;
        opacity: 0.95;
    ">
        ML Disease Screening &nbsp;•&nbsp;
        Symptom Analysis &nbsp;•&nbsp;
        Preventive Guidance &nbsp;•&nbsp;
        AI Health Chatbot
    </div>

</div>
""")


# ============================================================
# LOAD ML MODEL
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"""
        ❌ Model file not found.

        Expected location:

        {MODEL_PATH}

        Make sure your project structure is:

        Ai_health_model/
        │
        ├── app.py
        │
        └── model/
            └── health_model.joblib
        """
    )

    st.stop()


try:

    bundle = joblib.load(MODEL_PATH)

except Exception as e:

    st.error(
        f"❌ Could not load health model: {e}"
    )

    st.stop()


# ============================================================
# READ MODEL BUNDLE
# ============================================================

try:

    model = bundle["model"]

    features = bundle["features"]

    classes = bundle["classes"]

    model_name = bundle.get(
        "model_name",
        "Machine Learning Model"
    )

except Exception as e:

    st.error(
        f"""
        ❌ The health_model.joblib file does not
        contain the expected model information.

        Error:
        {e}
        """
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🩺 MediAI")

    st.write(
        "AI-Powered Health Assistant"
    )

    st.divider()

    st.write(
        "**ML Model:**"
    )

    st.info(
        model_name
    )

    st.write(
        f"**Symptoms:** {len(features)}"
    )

    st.write(
        f"**Disease Classes:** {len(classes)}"
    )

    st.divider()

    st.info(
        """
        Select symptoms in the
        Disease Screening tab and
        use the AI Health Chatbot
        for general health education.
        """
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 Disease Screening",
        "🤖 AI Health Chatbot",
        "📊 Model Details"
    ]
)


# ============================================================
# TAB 1 — DISEASE SCREENING
# ============================================================

with tab1:

    st.header(
        "🔎 Symptom-Based Disease Screening"
    )

    st.write(
        "Select the symptoms currently present "
        "and run the machine-learning prediction."
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=18,
            step=1
        )

    with col2:

        duration = st.number_input(
            "Symptom duration (days)",
            min_value=0,
            max_value=365,
            value=1,
            step=1
        )

    st.caption(
        "Age and duration are collected as basic of our data."
    )

    st.divider()

    # --------------------------------------------------------
    # SYMPTOM SEARCH
    # --------------------------------------------------------

    st.subheader(
        "🩺 Select Symptoms"
    )

    search = st.text_input(
        "🔍 Search for a symptom",
        placeholder="Example: fever, cough, headache..."
    )

    if search.strip():

        visible_features = [
            feature
            for feature in features
            if search.lower() in feature.lower()
        ]

    else:

        visible_features = features


    # --------------------------------------------------------
    # SYMPTOM CHECKBOXES
    # --------------------------------------------------------

    selected_symptoms = []

    symptom_columns = st.columns(3)

    for i, feature in enumerate(
        visible_features
    ):

        label = (
            str(feature)
            .replace("_", " ")
            .title()
        )

        checked = symptom_columns[
            i % 3
        ].checkbox(
            label,
            key=f"symptom_{feature}"
        )

        if checked:

            selected_symptoms.append(
                feature
            )


    # --------------------------------------------------------
    # SELECTED SYMPTOM COUNT
    # --------------------------------------------------------

    st.write(
        f"**Selected symptoms:** "
        f"{len(selected_symptoms)}"
    )

    st.divider()


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Analyze Symptoms",
        type="primary",
        use_container_width=True,
        key="predict_button"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        if not selected_symptoms:

            st.warning(
                "⚠️ Please select at least one symptom."
            )

        else:

            try:

                # ------------------------------------------------
                # CREATE INPUT VECTOR
                # ------------------------------------------------

                input_values = []

                for feature in features:

                    if feature in selected_symptoms:

                        input_values.append(1)

                    else:

                        input_values.append(0)


                input_data = pd.DataFrame(
                    [input_values],
                    columns=features
                )


                # ------------------------------------------------
                # MODEL PREDICTION
                # ------------------------------------------------

                prediction = model.predict(
                    input_data
                )[0]


                # ------------------------------------------------
                # PROBABILITIES
                # ------------------------------------------------

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            input_data
                        )[0]
                    )

                    model_classes = (
                        model.classes_
                    )

                    top_indices = np.argsort(
                        probabilities
                    )[::-1][:5]


                    result_df = pd.DataFrame(
                        {
                            "Disease": [
                                str(
                                    model_classes[i]
                                )
                                for i in top_indices
                            ],

                            "Probability": [
                                float(
                                    probabilities[i]
                                )
                                for i in top_indices
                            ]
                        }
                    )

                else:

                    result_df = pd.DataFrame(
                        {
                            "Disease":
                            [str(prediction)],

                            "Probability":
                            [1.0]
                        }
                    )


                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.success(
                    "✅ Analysis completed."
                )

                st.subheader(
                    "🩺 Screening Result"
                )

                st.metric(
                    "Top Predicted Class",
                    str(prediction)
                )


                # ------------------------------------------------
                # PROBABILITY CHART
                # ------------------------------------------------

                st.subheader(
                    "📊 Prediction Probabilities"
                )

                chart = px.bar(
                    result_df.sort_values(
                        "Probability"
                    ),
                    x="Probability",
                    y="Disease",
                    orientation="h",
                    range_x=[0, 1],
                    text="Probability",
                    title="Top Prediction Results"
                )

                chart.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition="outside"
                )

                chart.update_layout(
                    xaxis_title="Probability",
                    yaxis_title="Disease",
                    height=400
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )


                # ------------------------------------------------
                # SELECTED SYMPTOMS
                # ------------------------------------------------

                st.subheader(
                    "✓ Selected Symptoms"
                )

                for symptom in selected_symptoms:

                    st.write(
                        f"✓ {str(symptom).replace('_', ' ').title()}"
                    )


                # ------------------------------------------------
                # GENERAL GUIDANCE
                # ------------------------------------------------

                st.subheader(
                    "💡 General Preventive Guidance"
                )

                st.info(
                    """
                    • Maintain good hygiene.

                    • Get adequate sleep.

                    • Maintain a balanced diet.

                    • Stay physically active.

                    • Drink adequate water.

                    • Follow appropriate preventive
                      healthcare practices.

                    If symptoms are severe, persistent,
                    rapidly worsening, or concerning,
                    seek help from a qualified healthcare
                    professional.
                    """
                )


                # ------------------------------------------------
                # MEDICAL DISCLAIMER
                # ------------------------------------------------

                st.markdown(
                    """
                    <div class="medical-warning">

                
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            except Exception as e:

                st.error(
                    f"❌ Prediction error: {e}"
                )


# ============================================================
# TAB 2 — EVIDENCEMD AI HEALTH CHATBOT
# ============================================================

with tab2:

    st.header("🤖 AI Health Chatbot")

    st.write(
        "Ask general health-related questions and "
        "get evidence-based educational responses."
    )

    # ========================================================
    # INFO BOX
    # ========================================================

    st.markdown(
        """
        <div style="
            background: linear-gradient(
                black,
                black,
                gray
            );
            padding: 22px;
            border-radius: 18px;
            border: 1px solid red;
            margin-bottom: 20px;
        ">     

        <h3>💬 Ask MediAI</h3>

        <p>
        Ask about general health education, symptoms,
        prevention, healthy habits, medical terminology,
        and other educational health topics.
        </p>

        <p style="color:;">
        🔎 Evidence-based responses may include
        references and sources.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    question = st.text_area(
        "🩺 Ask your health question",
        placeholder=(
            "Example: What are common ways to reduce "
            "the risk of seasonal infections?"
        ),
        height=130,
        key="evidencemd_health_question"
    )


    # ========================================================
    # ASK BUTTON
    # ========================================================

    ask_button = st.button(
        "🤖 Ask MediAI",
        type="primary",
        use_container_width=True,
        key="evidencemd_ask_button"
    )


    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if ask_button:

        # ----------------------------------------------------
        # CHECK QUESTION
        # ----------------------------------------------------

        if not question.strip():

            st.warning(
                "⚠️ Please enter a health question first."
            )

        # ----------------------------------------------------
        # CHECK API KEY
        # ----------------------------------------------------

        elif not EVIDENCEMD_API_KEY:

            st.error(
                "❌ EvidenceMD API key is missing."
            )

            st.info(
                "Add EVIDENCEMD_API_KEY to your .env file."
            )

        # ----------------------------------------------------
        # SEND REQUEST
        # ----------------------------------------------------

        else:

            try:

                # =================================================
                # HEADERS
                # =================================================

                headers = {
                    "x-api-key": EVIDENCEMD_API_KEY,
                    "Content-Type": "application/json"
                }


                # =================================================
                # REQUEST PAYLOAD
                # =================================================

                payload = {

                    "model": EVIDENCEMD_MODEL,

                    "messages": [

                        {
                            "role": "system",

                            "content": """
You are MediAI, a health education assistant.

Your purpose is to provide general,
easy-to-understand health information.

Important safety rules:

1. Do not provide a confirmed diagnosis.

2. Do not tell the user that they definitely
have a disease.

3. Do not prescribe medicines or provide
medication dosage instructions.

4. Explain uncertainty when discussing symptoms.

5. If a user describes symptoms that could
require urgent professional attention,
recommend seeking appropriate medical care.

6. Keep answers clear and educational.

7. Do not request unnecessary personal information.

8. Clearly explain that the response is
educational information and is not a substitute
for a qualified healthcare professional.

9. Do not present AI output as a medical diagnosis.
"""
                        },

                        {
                            "role": "user",

                            "content": question.strip()
                        }
                    ],

                    "temperature": 0.7,

                    "max_tokens": 1000
                }


                # =================================================
                # API REQUEST
                # =================================================

                with st.spinner(
                    "🤖 MediAI is searching for an evidence-based answer..."
                ):

                    response = requests.post(
                        EVIDENCEMD_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )


                # =================================================
                # SUCCESS
                # =================================================

                if response.status_code == 200:

                    data = response.json()


                    # -------------------------------------------------
                    # GET ANSWER
                    # -------------------------------------------------

                    answer = ""

                    choices = data.get(
                        "choices",
                        []
                    )

                    if choices:

                        message = choices[0].get(
                            "message",
                            {}
                        )

                        answer = message.get(
                            "content",
                            ""
                        )


                    # -------------------------------------------------
                    # DISPLAY ANSWER
                    # -------------------------------------------------

                    if answer:

                        st.subheader(
                            "🤖 MediAI Response"
                        )

                        st.markdown(
                            """
                            <div style="
                                background:#ffffff;
                                padding:24px;
                                border-radius:18px;
                                border:1px solid #dce7f2;
                                box-shadow:0 4px 15px rgba(0,0,0,0.06);
                                margin-top:10px;
                                line-height:1.7;
                            ">
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(answer)

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )


                        # =================================================
                        # SOURCES / EVIDENCE
                        # =================================================

                        sources = data.get(
                            "sources",
                            []
                        )


                        # Sometimes sources may be inside
                        # the message object.
                        if not sources and choices:

                            message = choices[0].get(
                                "message",
                                {}
                            )

                            sources = message.get(
                                "sources",
                                []
                            )


                        if sources:

                            st.subheader(
                                "📚 Evidence & Sources"
                            )

                            for i, source in enumerate(
                                sources,
                                start=1
                            ):

                                if isinstance(
                                    source,
                                    dict
                                ):

                                    title = source.get(
                                        "title",
                                        f"Source {i}"
                                    )

                                    url = source.get(
                                        "url",
                                        ""
                                    )

                                    journal = source.get(
                                        "journal",
                                        ""
                                    )

                                    year = source.get(
                                        "year",
                                        ""
                                    )

                                    st.markdown(
                                        f"**[{i}] {title}**"
                                    )

                                    if journal:

                                        st.caption(
                                            f"Journal: {journal}"
                                        )

                                    if year:

                                        st.caption(
                                            f"Year: {year}"
                                        )

                                    if url:

                                        st.markdown(
                                            f"[🔗 View Source]({url})"
                                        )

                                else:

                                    st.markdown(
                                        f"**[{i}] {source}**"
                                    )


                        # =================================================
                        # RAW SOURCES FALLBACK
                        # =================================================

                        elif data.get("sources"):

                            st.subheader(
                                "📚 Evidence & Sources"
                            )

                            st.json(
                                data.get("sources")
                            )


                        # =================================================
                        # DISCLAIMER
                        # =================================================

                        st.info(
                            """
                            ⚠️ **Educational information only**

                            This AI response is not a medical diagnosis
                            or a substitute for advice from a qualified
                            healthcare professional.
                            """
                        )


                    else:

                        st.error(
                            "❌ EvidenceMD returned an empty response."
                        )

                        st.json(data)


                # =================================================
                # AUTHENTICATION ERROR
                # =================================================

                elif response.status_code in [401, 403]:

                    st.error(
                        "❌ EvidenceMD API authentication failed."
                    )

                    st.info(
                        """
                        Check that your EVIDENCEMD_API_KEY
                        in the .env file is correct and active.
                        """
                    )


                # =================================================
                # RATE LIMIT
                # =================================================

                elif response.status_code == 429:

                    st.error(
                        "❌ EvidenceMD API rate limit reached."
                    )

                    st.info(
                        "Please wait and try again."
                    )


                # =================================================
                # SERVER ERROR
                # =================================================

                elif response.status_code >= 500:

                    st.error(
                        "❌ EvidenceMD server error."
                    )

                    st.info(
                        "Please try again after some time."
                    )


                # =================================================
                # OTHER API ERROR
                # =================================================

                else:

                    st.error(
                        f"❌ EvidenceMD API Error: "
                        f"{response.status_code}"
                    )

                    try:

                        error_data = response.json()

                        st.json(error_data)

                    except Exception:

                        st.code(
                            response.text
                        )


            # =====================================================
            # TIMEOUT
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "❌ Request timed out. "
                    "Please try again."
                )


            # =====================================================
            # CONNECTION ERROR
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to EvidenceMD."
                )

                st.info(
                    "Check your internet connection."
                )


            # =====================================================
            # OTHER ERROR
            # =====================================================

            except Exception as e:

                st.error(
                    f"❌ Unexpected error: {e}"
                )


# ============================================================
# TAB 3 — MODEL DETAILS
# ============================================================

with tab3:

    st.header(
        "📊 Model Details"
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Features",
            len(features)
        )

    with col2:

        st.metric(
            "Disease Classes",
            len(classes)
        )


    st.write(
        "**Selected ML Model:**"
    )

    st.info(
        model_name
    )


    # --------------------------------------------------------
    # DISEASE CLASSES
    # --------------------------------------------------------

    st.subheader(
        "🩺 Disease Classes"
    )

    class_df = pd.DataFrame(
        {
            "Disease": [
                str(x)
                for x in classes
            ]
        }
    )

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader(
        "🔬 Model Features"
    )

    feature_df = pd.DataFrame(
        {
            "Feature": [
                str(x)
                for x in features
            ]
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # PROJECT PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "🚀 Project Pipeline"
    )

    st.code(
        """
External Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
X / y Separation
       ↓
Stratified 80/20 Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model
       ↓
health_model.joblib
       ↓
Streamlit Application
       ↓
User Selects Symptoms
       ↓
ML Disease Screening
       ↓
DR7.ai Medical AI Chatbot
        """
    )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown(
        """
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 AI Health Assistant"
)

    