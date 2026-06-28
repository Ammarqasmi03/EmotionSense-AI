import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INLINE MODERN THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external or premium embedded CSS fallback
def load_css(file_name="style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # High-end Dark Gradient Theme fallback
        st.markdown("""
            <style>
                /* Hero Section Styling */
                .hero-section { 
                    text-align: center; 
                    padding: 2.5rem 1.5rem; 
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(168, 85, 247, 0.1));
                    border-radius: 16px;
                    border: 1px solid rgba(168, 85, 247, 0.2);
                    margin-bottom: 2.5rem;
                    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                }
                .hero-section h1 {
                    background: linear-gradient(45deg, #3b82f6, #a855f7);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 800;
                }
                
                /* Glassmorphism Metric Cards */
                .metric-card { 
                    background: rgba(30, 41, 59, 0.45); 
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    border: 1px solid rgba(255, 255, 255, 0.08); 
                    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
                    margin-bottom: 1rem;
                }
                
                /* Streamlit default element overrides for custom feel */
                div.stButton > button:first-child {
                    background: linear-gradient(90deg, #3b82f6, #a855f7) !important;
                    color: white !important;
                    border: none !important;
                    font-weight: bold !important;
                    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
                }
                div.stButton > button:first-child:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 0 15px rgba(168, 85, 247, 0.5) !important;
                }
                
                /* Fine-tuned Footer */
                .footer { 
                    text-align: center; 
                    margin-top: 4rem; 
                    padding: 2rem 0; 
                    border-top: 1px solid rgba(255, 255, 255, 0.05); 
                }
                .footer a {
                    color: #3b82f6 !important;
                    font-weight: 600;
                    transition: color 0.2s ease;
                }
                .footer a:hover {
                    color: #a855f7 !important;
                }
            </style>
        """, unsafe_allow_html=True)

load_css("style.css")

# -----------------------------------------------------------------------------
# 2. MODEL & UTILITY LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_components():
    """Loads and caches the trained model, vectorizer, and label encoder."""
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        st.error("⚠️ Model or Vectorizer files not found! Please ensure 'model.pkl', 'vectorizer.pkl' and 'label_encoder.pkl' are in the project root.")
        return None, None, None

model, vectorizer, label_encoder = load_ml_components()

# Emoji Mapping Constants
EMOJI_MAPPING = {
    "Anger": "😡",
    "Fear": "😨",
    "Joy": "😊",
    "Love": "❤️",
    "Sadness": "😢",
    "Surprise": "😲"
}

# Initialize Session State for History
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -----------------------------------------------------------------------------
# 3. CORE LOGIC FUNCTIONS
# -----------------------------------------------------------------------------
def predict_emotion(text, model, vectorizer, label_encoder):
    """Transforms input text and returns predicted class, confidence, and sorted probabilities."""
    # Transform text using the loaded CountVectorizer
    transformed_text = vectorizer.transform([text])
    
    # Predict probabilities and classes
    probabilities = model.predict_proba(transformed_text)[0]
    
    # Decoded & capitalized classes map smoothly to presentation layers
    classes = [str(emotion).capitalize() for emotion in label_encoder.inverse_transform(model.classes_)]
    
    # Create a clean dictionary mapping
    prob_map = {classes[i]: probabilities[i] for i in range(len(classes))}
    
    # Sort probabilities from highest to lowest
    sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    
    top_emotion = sorted_probs[0][0]
    top_confidence = sorted_probs[0][1]
    
    return top_emotion, top_confidence, sorted_probs

def add_to_history(text, emotion):
    """Maintains a rolling log of the last 5 unique or newest predictions."""
    preview_text = text if len(text) <= 40 else text[:37] + "..."
    new_entry = {"text": f'"{preview_text}"', "emotion": emotion}
    
    st.session_state.prediction_history.insert(0, new_entry)
    
    # Keep only the latest 5 entries
    if len(st.session_state.prediction_history) > 5:
        st.session_state.prediction_history = st.session_state.prediction_history[:5]

# -----------------------------------------------------------------------------
# 4. SIDEBAR IMPLEMENTATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center; background: linear-gradient(45deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🧠 EmotionSense AI</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📋 About Project")
    st.caption("An intelligent NLP application capable of identifying underlying emotional sentiments within text inputs in real time.")
    
    st.markdown("### 📊 Model Metrics")
    st.markdown("**Algorithm:** `Logistic Regression`")
    st.markdown("**Feature Extraction:** `CountVectorizer`")
    st.markdown("**Accuracy:** <span style='color: #2ecc71; font-weight: bold;'>90%</span>", unsafe_allow_html=True)
    
    st.markdown("### 📂 Dataset Information")
    st.caption("Trained on a multiclass English Emotion Dataset classifying texts into Joy, Sadness, Anger, Fear, Love, and Surprise.")
    
    st.markdown("### 🛠️ Technologies Used")
    st.markdown(
        "• Python  \n• Streamlit  \n• Scikit-Learn  \n• Pandas  \n• NumPy"
    )
    st.markdown("---")

# -----------------------------------------------------------------------------
# 5. MAIN HERO SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-section">
        <h1 style="font-size: 2.8rem; margin-bottom: 0.5rem;">😊 EmotionSense AI</h1>
        <h3 style="font-weight: 400; opacity: 0.9; margin-top: 0;">AI-Powered Emotion Detection System</h3>
        <p style="font-size: 1.1rem; opacity: 0.75; max-width: 700px; margin: 0 auto;">
            Analyze the emotion behind any sentence using Machine Learning and Natural Language Processing.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 6. MAIN BODY LAYOUT
# -----------------------------------------------------------------------------
col1, col2 = st.columns([5, 4], gap="large")

with col1:
    st.markdown("### 📝 Input Analytics Area")
    user_input = st.text_area(
        label="Enter your text sentences below:",
        placeholder="I am feeling very excited for my interview tomorrow.",
        height=150,
        label_visibility="collapsed"
    )
    
    # Large execution button layout
    btn_space1, btn_space2, btn_space3 = st.columns([1, 2, 1])
    with btn_space2:
        predict_btn = st.button("Detect Emotion", use_container_width=True, type="primary")

    # Action Execution Trigger
    if predict_btn:
        if not user_input.strip():
            st.error("🚨 Please type a valid sentence before analyzing!")
        elif model is None or vectorizer is None or label_encoder is None:
            st.error("🚨 Core engine missing. Check backend files.")
        else:
            with st.spinner("Analyzing emotional dimensions..."):
                # Compute Prediction
                emotion, confidence, sorted_probabilities = predict_emotion(user_input, model, vectorizer, label_encoder)
                
                # Cache results to history
                add_to_history(user_input, emotion)
                
                st.session_state.current_results = {
                    "emotion": emotion,
                    "confidence": confidence,
                    "probabilities": sorted_probabilities,
                    "triggered": True
                }
                st.balloons()

    # Persistent Display Area for current results
    if "current_results" in st.session_state and st.session_state.current_results.get("triggered"):
        res = st.session_state.current_results
        emoji = EMOJI_MAPPING.get(res["emotion"], "✨")
        
        st.markdown("### 🏆 Prediction Summary")
        st.markdown(
            f"""
            <div class="metric-card" style="text-align: center; border: 1px solid rgba(168, 85, 247, 0.3);">
                <h1 style="font-size: 3.8rem; margin-bottom: 0; filter: drop-shadow(0 0 10px rgba(168,85,247,0.3));">{emoji} {res["emotion"]}</h1>
                <p style="letter-spacing: 1px; font-size: 0.9rem; text-transform: uppercase; opacity: 0.6; margin-top: 5px;">Predicted Emotion</p>
                <hr style="margin: 15px 0; opacity: 0.1;">
                <h2 style="margin: 0; color: #a855f7; font-size: 2.5rem;">{res["confidence"]*100:.1f}%</h2>
                <p style="font-size: 0.85rem; opacity: 0.6; margin: 0;">Confidence Score</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

with col2:
    st.markdown("### 📊 Probability Breakdown")
    if "current_results" in st.session_state and st.session_state.current_results.get("triggered"):
        for cls_name, prob_val in st.session_state.current_results["probabilities"]:
            cls_emoji = EMOJI_MAPPING.get(cls_name, "▪️")
            percentage = prob_val * 100
            
            st.markdown(f"**{cls_emoji} {cls_name}** • {percentage:.1f}%")
            st.progress(prob_val)
    else:
        st.info("Complete an emotion prediction check to populate live breakdown analytics here.")

st.markdown("<br><hr style='opacity: 0.1;'><br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. INFORMATION CARDS & HISTORY
# -----------------------------------------------------------------------------
col_info, col_hist = st.columns([5, 4], gap="large")

with col_info:
    st.markdown("### ℹ️ About Project")
    st.markdown(
        """
        <div class="metric-card">
            <p style="line-height: 1.6; margin: 0;"><strong>EmotionSense AI</strong> is an NLP-based emotion detection application developed using 
            Streamlit and Scikit-Learn. It predicts the emotion behind user-entered text using 
            <code style="color:#3b82f6;">CountVectorizer</code> and a <code style="color:#a855f7;">Logistic Regression</code> classifier trained seamlessly on a 
            multiclass emotion classification dataset.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("### ⚙️ Model Details")
    df_meta = pd.DataFrame({
        "Parameter": ["Algorithm", "Vectorizer", "Accuracy Target", "Language Framework"],
        "Value": ["Logistic Regression", "CountVectorizer", "90%", "English (EN)"]
    })
    st.table(df_meta.set_index("Parameter"))

with col_hist:
    st.markdown("### ⏳ Recent Predictions")
    if st.session_state.prediction_history:
        for item in st.session_state.prediction_history:
            hist_emoji = EMOJI_MAPPING.get(item['emotion'], "✨")
            st.markdown(
                f"""
                <div style="padding: 10px 15px; background: rgba(255,255,255,0.02); border-left: 4px solid #3b82f6; margin-bottom: 10px; border-radius: 0 8px 8px 0; border-top: 1px solid rgba(255,255,255,0.03); border-right: 1px solid rgba(255,255,255,0.03);">
                    <span style="font-style: italic; opacity: 0.85;">{item['text']}</span> <br>
                    <strong style="color: #a855f7; font-size: 0.95rem;">&rarr; {hist_emoji} {item['emotion']}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.caption("No recent history queries captured yet in this current session workflow context.")

# -----------------------------------------------------------------------------
# 8. FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <p style="margin-bottom: 5px; font-weight: bold; font-size: 1.1rem;">Developed by Ammar Gour</p>
        <p style="margin-top: 0; font-size: 0.9rem; opacity: 0.6; margin-bottom: 15px;">MCA (AI & ML) | Jamia Millia Islamia</p>
        <p style="font-size: 1.1rem;">
            🔗 <a href="https://github.com/Ammarqasmi03/EmotionSense-AI" target="_blank" style="text-decoration:none;">GitHub</a> &nbsp;|&nbsp; 
            🔗 <a href="https://www.linkedin.com/in/ammar-qasmi-082266289" target="_blank" style="text-decoration:none;">LinkedIn</a> &nbsp;|&nbsp; 
            🔗 <a href="https://ammarqasmi03.github.io/my-portfilio/" target="_blank" style="text-decoration:none;">Portfolio</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)