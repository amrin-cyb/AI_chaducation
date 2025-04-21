import streamlit as st
from transformers import pipeline

# --- Title & Description ---
st.set_page_config(page_title="Healthy Lifestyle AI Coach", page_icon="💪")
st.title("🧠 Healthy Lifestyle AI Coach")
st.markdown("Ask me anything about how to live a healthier life!")

# --- Define the Health Knowledge Base ---
healthy_lifestyle_text = """
Living a healthy lifestyle involves these key habits:

1. Physical Activity:
   - Exercise for at least 30 minutes daily (e.g., walking, jogging, home workouts).
   - Stay active throughout the day to improve heart health and fitness.

2. Balanced Diet:
   - Eat a variety of fruits, vegetables, whole grains, and lean proteins.
   - Reduce sugar, salt, and processed foods.
   - Stay hydrated by drinking plenty of water.

3. Sleep & Recovery:
   - Aim for 7–9 hours of quality sleep each night.
   - Maintain a regular sleep schedule.

4. Mental Wellness:
   - Manage stress through meditation, hobbies, or journaling.
   - Spend time with supportive family and friends.
   - Take breaks and avoid burnout.

5. Healthy Choices:
   - Avoid smoking and limit alcohol consumption.
   - Practice good hygiene and regular health checkups.

These habits together support long-term physical and mental well-being.
"""

# --- Load Hugging Face QA Pipeline ---
@st.cache_resource
def load_model():
    return pipeline("question-answering")

qa_pipeline = load_model()

# --- User Input ---
user_question = st.text_input("📝 Enter your question:")

# --- Generate and Show Answer ---
if user_question:
    with st.spinner("Thinking... 💭"):
        result = qa_pipeline(question=user_question, context=healthy_lifestyle_text)
        st.markdown("**💬 AI Coach says:**")
        st.success(result["answer"])
