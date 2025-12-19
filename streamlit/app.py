# streamlit/app.py
"""
Streamlit frontend for Titanic Survival Prediction.
"""

import streamlit as st
import requests
import json

# Configuration
API_URL = "http://api:8000"

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Header
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Will you survive the Titanic?")
st.markdown("---")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts whether a passenger would survive the Titanic disaster.
    
    **Model:** Random Forest  
    **F1-Score:** 0.7385  
    **Accuracy:** 81.01%
    
    Enter passenger information and click **Predict Survival** to see the result!
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")

# Main form
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Information")
    
    name = st.text_input("Passenger Name", value="John Doe", help="For display only")
    
    pclass = st.selectbox(
        "Ticket Class",
        options=[1, 2, 3],
        format_func=lambda x: f"Class {x} ({'1st' if x==1 else '2nd' if x==2 else '3rd'})",
        help="1 = First Class, 2 = Second Class, 3 = Third Class"
    )
    
    sex = st.radio("Sex", options=["male", "female"], horizontal=True)
    
    age = st.slider("Age", min_value=0, max_value=80, value=30, step=1)

with col2:
    st.subheader("👨‍👩‍👧‍👦 Family & Fare")
    
    siblings_spouses = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0,
        help="Number of siblings or spouses aboard"
    )
    
    parents_children = st.number_input(
        "Parents/Children Aboard",
        min_value=0,
        max_value=6,
        value=0,
        help="Number of parents or children aboard"
    )
    
    fare = st.number_input(
        "Fare Paid (£)",
        min_value=0.0,
        max_value=500.0,
        value=32.0,
        step=0.5,
        help="Ticket fare in British Pounds"
    )
    
    port_code = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x]
    )

# Predict button
st.markdown("---")

if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
    # Prepare request
    payload = {
        "instances": [
            {
                "pclass": int(pclass),
                "sex": sex,
                "age": float(age),
                "siblings_spouses": int(siblings_spouses),
                "parents_children": int(parents_children),
                "fare": float(fare),
                "port_code": port_code
            }
        ]
    }
    
    # Make prediction
    with st.spinner("🔮 Predicting..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["predictions"][0]
                probability = result["probabilities"][0]
                
                st.markdown("---")
                st.markdown("## 🎯 Prediction Result")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if prediction == "Survived":
                        st.success(f"### ✅ {name} would have **SURVIVED!**")
                        st.metric(
                            "Survival Probability",
                            f"{probability*100:.1f}%",
                            delta=f"{(probability-0.5)*100:.1f}% vs average"
                        )
                        st.balloons()
                    else:
                        st.error(f"### ❌ {name} would **NOT** have survived")
                        st.metric(
                            "Survival Probability",
                            f"{probability*100:.1f}%",
                            delta=f"{(probability-0.5)*100:.1f}% vs average"
                        )
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    st.json(payload["instances"][0])
                
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.json(response.json())
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Could not connect to API. Is the service running?")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Titanic Survival Prediction | ML Model: Random Forest | F1-Score: 0.7385</p>
    <p>Built with FastAPI + Streamlit | Deployed on DigitalOcean</p>
</div>
""", unsafe_allow_html=True)
