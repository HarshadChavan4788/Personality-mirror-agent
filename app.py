import streamlit as st
import os
from crew_agents import run_multiagent_personality_pipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Personality Mirror — MultiAgent (CrewAI)",
    layout="centered"
)

# ---------------------------------------------------------
# LIGHT PINK + CLEAN CARD UI
# ---------------------------------------------------------
st.markdown("""
<style>
    body, .stApp {
        background: #fdeef4 !important;
    }
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    .trait-box {
        background: white;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #ff9ebd;
    }
    h2, h3 {
        color: #333;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #ff7fa8 !important;
        color: white !important;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #ff5f94 !important;
    }
    .stTextInput>div>input {
        background: white !important;
        border-radius: 8px !important;
        border: 1px solid #f3c6d5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🪞 Personality Mirror")
st.write("A soft, friendly multi-agent personality analysis powered by CrewAI + Gemini.")

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
with st.form("personality_form"):
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Your name (optional)")
        q1 = st.text_input("1) What do you enjoy doing in your free time?")
        q2 = st.text_input("2) How would your friends describe you in 3 words?")

    with col2:
        q3 = st.text_input("3) What stresses you out the most?")
        q4 = st.text_input("4) Describe a recent decision you made and why.")
        q5 = st.text_input("5) What is a personal strength and a weakness?")

    submitted = st.form_submit_button("Generate Personality Mirror")

# ---------------------------------------------------------
# Run Multi-Agent Pipeline
# ---------------------------------------------------------
if submitted:
    answers = [q1.strip(), q2.strip(), q3.strip(), q4.strip(), q5.strip()]

    if not any(answers):
        st.error("Please answer at least one question.")
    else:
        with st.spinner("Analyzing your personality..."):
            try:
                result = run_multiagent_personality_pipeline(answers, name=name)
                st.success("Your personality mirror is ready!")

                # ---------------------------------------------------------
                # Summary
                # ---------------------------------------------------------
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("Summary")
                st.write(result.get("summary", "—"))
                st.markdown("</div>", unsafe_allow_html=True)

                # ---------------------------------------------------------
                # Traits
                # ---------------------------------------------------------
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("Traits")

                traits = result.get("traits", {})
                for trait, score in traits.items():
                    st.markdown(
                        f"""
                        <div class='trait-box'>
                            <b>{trait}</b><br>
                            Score: {score}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                # ---------------------------------------------------------
                # Recommendations
                # ---------------------------------------------------------
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("Recommendations")

                for rec in result.get("recommendations", []):
                    st.write(f"- {rec}")

                st.markdown("</div>", unsafe_allow_html=True)

                # ---------------------------------------------------------
                # Validation Message
                # ---------------------------------------------------------
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("Validation")
                st.write(result.get("validating_message", ""))
                st.markdown("</div>", unsafe_allow_html=True)

                # ---------------------------------------------------------
                # Download Button
                # ---------------------------------------------------------
                st.download_button(
                    label="Download Report",
                    data=str(result),
                    file_name="personality_mirror_report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error generating personality mirror: {e}")
