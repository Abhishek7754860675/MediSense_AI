import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

st.set_page_config(page_title="MediSense AI", page_icon="🩺", layout="centered")

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #0f766e 0%, #0891b2 100%);
    padding: 2rem; border-radius: 16px; margin-bottom: 1.5rem; text-align: center;
}
.main-header h1 { color: white; margin: 0; font-size: 2.2rem; }
.main-header p { color: #e0f2fe; margin: 0.3rem 0 0 0; font-size: 1rem; }
.option-card {
    background: #ffffff; border: 2px solid #e2e8f0; border-radius: 16px;
    padding: 1.5rem; text-align: center; height: 100%;
}
.option-card h3 { margin-top: 0.5rem; }
.result-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; margin-top: 1rem;
}
.urgency-high { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
.urgency-medium { background: #fef3c7; color: #ca8a04; border: 1px solid #fde68a; }
.urgency-low { background: #dcfce7; color: #16a34a; border: 1px solid #bbf7d0; }
.urgency-badge {
    display: inline-block; padding: 0.4rem 1.2rem; border-radius: 20px; font-weight: 700; margin-top: 0.6rem;
}
.symptom-chip {
    display: inline-block; background: #f0fdfa; color: #0f766e; border: 1px solid #99f6e4;
    border-radius: 20px; padding: 0.3rem 0.9rem; margin: 0.2rem; font-size: 0.85rem;
}
.disclaimer-box {
    background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 8px;
    padding: 1rem; margin-top: 1.2rem; text-align: left;
}
.info-box {
    background: #f0f9ff; border-left: 4px solid #0891b2; border-radius: 8px;
    padding: 1rem; margin-top: 1rem; text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🩺 MediSense AI</h1>
    <p>AI Health Diagnosis Assistant</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session state (controls which "page" is showing)
# ------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "home"      # home -> input -> result -> hospital
if "flow" not in st.session_state:
    st.session_state.flow = None         # "symptom" or "skin"
if "prediction" not in st.session_state:
    st.session_state.prediction = None   # (disease, confidence, symptom_count)
if "hospitals" not in st.session_state:
    st.session_state.hospitals = []

def go_home():
    st.session_state.stage = "home"
    st.session_state.flow = None
    st.session_state.prediction = None

# ------------------------------------------------------------------
# Shared helpers — Urgency (Module 3) + brief disease info
# ------------------------------------------------------------------
@st.cache_resource
def load_severity_map():
    if os.path.exists("severity_map.pkl"):
        return joblib.load("severity_map.pkl")
    return {}

severity_map = load_severity_map()

def calculate_urgency(disease, confidence, symptom_count):
    severity = severity_map.get(disease, 5)
    severity_pct = severity * 10
    confidence_pct = confidence
    symptom_pct = min(symptom_count / 10, 1) * 100
    score = round((0.4 * severity_pct) + (0.3 * confidence_pct) + (0.3 * symptom_pct), 2)
    category = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
    return score, category

def urgency_badge(category):
    cls = {"High": "urgency-high", "Medium": "urgency-medium", "Low": "urgency-low"}[category]
    return f'<span class="urgency-badge {cls}">Urgency: {category}</span>'

# A few common one-line notes — kept general/educational, not medical advice.
# Anything not listed falls back to a generic message.
DISEASE_INFO = {
    "Common Cold": "A mild viral infection of the nose and throat.",
    "Fungal infection": "A skin condition caused by fungal overgrowth, often in warm/moist areas.",
    "Migraine": "A neurological condition causing recurring, often one-sided headaches.",
    "GERD": "Acid reflux condition where stomach acid irritates the food pipe.",
    "Diabetes ": "A metabolic condition affecting how the body regulates blood sugar.",
    "Hypertension ": "Persistently high blood pressure in the arteries.",
    "Pneumonia": "An infection that inflames the air sacs in one or both lungs.",
    "Typhoid": "A bacterial infection spread through contaminated food or water.",
    "Malaria": "A mosquito-borne infection caused by a parasite.",
    "Dengue": "A mosquito-borne viral infection causing fever and body aches.",
    "Chicken pox": "A viral infection causing an itchy, blister-like rash.",
    "Acne": "A common skin condition caused by clogged hair follicles.",
    "Melanoma": "A serious form of skin cancer that develops in pigment-producing cells.",
    "Melanocytic Nevi": "Common moles — usually harmless pigmented skin growths.",
    "Basal Cell Carcinoma": "The most common, usually slow-growing type of skin cancer.",
    "Benign Keratosis": "A non-cancerous skin growth, common with age.",
}

def get_disease_info(disease):
    return DISEASE_INFO.get(
        disease,
        "A general medical condition. Please consult a qualified doctor for a full explanation and confirmation."
    )

# ------------------------------------------------------------------
# Hospital search (Module 4, free — OpenStreetMap, with mirror fallback)
# ------------------------------------------------------------------
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

def find_nearby_hospitals(lat, lng, radius=8000):
    import requests
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius},{lat},{lng});
      way["amenity"="hospital"](around:{radius},{lat},{lng});
      node["amenity"="clinic"](around:{radius},{lat},{lng});
    );
    out center;
    """
    last_error = None
    for mirror in OVERPASS_MIRRORS:
        try:
            response = requests.post(
                mirror, data={"data": query}, timeout=30,
                headers={"User-Agent": "medisense_ai_app"}
            )
            if response.status_code != 200:
                last_error = f"{mirror} returned status {response.status_code}"
                continue
            elements = response.json().get("elements", [])
            hospitals = []
            for el in elements:
                name = el.get("tags", {}).get("name", "Unnamed Hospital")
                if el["type"] == "node":
                    h_lat, h_lng = el["lat"], el["lon"]
                else:
                    h_lat, h_lng = el["center"]["lat"], el["center"]["lon"]
                hospitals.append({"name": name, "lat": h_lat, "lng": h_lng})
            return hospitals, None   # success — even if hospitals is genuinely empty
        except Exception as e:
            last_error = f"{mirror} failed: {e}"
            time.sleep(1)
            continue

    return [], last_error   # all mirrors failed — return the real reason

# ====================================================================
# STAGE: HOME — 2 options with a short guide
# ====================================================================
if st.session_state.stage == "home":
    st.write("Choose how you'd like to check your health:")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="option-card">
            <div style="font-size:2.5rem;">🩹</div>
            <h3>Symptom Checker</h3>
            <p style="color:#64748b;">Select the symptoms you're feeling — get a predicted condition,
            confidence score, and urgency level.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Symptom Checker", use_container_width=True, type="primary"):
            st.session_state.stage = "input"
            st.session_state.flow = "symptom"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="option-card">
            <div style="font-size:2.5rem;">🔬</div>
            <h3>Skin Checker</h3>
            <p style="color:#64748b;">Upload a photo of a skin concern — get an AI-predicted condition
            with a Grad-CAM explanation.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Skin Checker", use_container_width=True, type="primary"):
            st.session_state.stage = "input"
            st.session_state.flow = "skin"
            st.rerun()

# ====================================================================
# STAGE: INPUT — symptom multiselect OR image upload
# ====================================================================
elif st.session_state.stage == "input":
    if st.button("← Back"):
        go_home()
        st.rerun()

    if st.session_state.flow == "symptom":
        st.markdown("#### 🩹 Select Your Symptoms")
        try:
            model = joblib.load("symptom_disease_model.pkl")
            encoder = joblib.load("label_encoder.pkl")
            symptom_columns = joblib.load("symptom_columns.pkl")

            def format_symptom(name):
                return name.replace("_", " ").title()

            display_to_raw = {format_symptom(s): s for s in symptom_columns}

            selected_display = st.multiselect(
                "symptoms", options=sorted(display_to_raw.keys()),
                placeholder="e.g. Headache, Fatigue, Joint Pain...", label_visibility="collapsed"
            )
            if selected_display:
                chips = "".join([f'<span class="symptom-chip">{s}</span>' for s in selected_display])
                st.markdown(chips, unsafe_allow_html=True)

            st.write("")
            if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
                if not selected_display:
                    st.warning("Please select at least one symptom.")
                else:
                    raw_symptoms = [display_to_raw[s] for s in selected_display]
                    input_vector = pd.DataFrame([np.zeros(len(symptom_columns))], columns=symptom_columns)
                    for s in raw_symptoms:
                        input_vector[s] = 1

                    probs = model.predict_proba(input_vector)[0]
                    top_idx = np.argsort(probs)[::-1][0]
                    disease = encoder.inverse_transform([top_idx])[0]
                    confidence = round(probs[top_idx] * 100, 2)

                    st.session_state.prediction = (disease, confidence, len(selected_display))
                    st.session_state.stage = "result"
                    st.rerun()

        except FileNotFoundError:
            st.error(
                "Model files not found. Make sure `symptom_disease_model.pkl`, `label_encoder.pkl`, "
                "and `symptom_columns.pkl` are in this same folder."
            )

    elif st.session_state.flow == "skin":
        st.markdown("#### 🔬 Upload a Skin Image")
        uploaded_image = st.file_uploader("image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_image is not None:
            try:
                from tensorflow.keras.models import load_model
                from PIL import Image

                @st.cache_resource
                def load_skin_artifacts():
                    model = load_model("skin_disease_model.keras")
                    class_indices = joblib.load("skin_class_indices.pkl")
                    idx_to_class = {v: k for k, v in class_indices.items()}
                    return model, idx_to_class

                skin_model, idx_to_class = load_skin_artifacts()

                img = Image.open(uploaded_image).convert("RGB").resize((224, 224))
                st.image(img, caption="Uploaded Image", use_container_width=True)

                if st.button("🔍 Predict Condition", type="primary", use_container_width=True):
                    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
                    preds = skin_model.predict(img_array, verbose=0)
                    pred_idx = int(np.argmax(preds))
                    disease = idx_to_class[pred_idx]
                    confidence = round(float(np.max(preds)) * 100, 2)

                    st.session_state.prediction = (disease, confidence, 1)
                    st.session_state.stage = "result"
                    st.rerun()

            except FileNotFoundError:
                st.error(
                    "Model files not found. Make sure `skin_disease_model.keras` and "
                    "`skin_class_indices.pkl` are in this same folder."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ====================================================================
# STAGE: RESULT — prediction + urgency + info + hospital button
# ====================================================================
elif st.session_state.stage == "result":
    disease, confidence, symptom_count = st.session_state.prediction
    score, category = calculate_urgency(disease, confidence, symptom_count)

    st.markdown(f"""
    <div class="result-card">
        <p style="color:#64748b; margin-bottom:0.2rem;">Predicted Condition</p>
        <h2 style="margin-top:0;">{disease}</h2>
        <p style="font-size:1.3rem; font-weight:700;">{confidence}% confidence</p>
        {urgency_badge(category)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        ℹ️ <b>About this condition:</b> {get_disease_info(disease)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <b>This is not a medical diagnosis.</b> Please consult a qualified doctor for proper evaluation.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    if st.button("🏥 Locate Best Hospital & Book Appointment", type="primary", use_container_width=True):
        st.session_state.stage = "hospital"
        st.rerun()

    if st.button("← Check Something Else"):
        go_home()
        st.rerun()

# ====================================================================
# STAGE: HOSPITAL — city search, map, demo booking
# ====================================================================
elif st.session_state.stage == "hospital":
    if st.button("← Back to Result"):
        st.session_state.stage = "result"
        st.rerun()

    st.markdown("#### 🏥 Find Nearby Hospitals")
    city = st.text_input("Enter your city (India)", placeholder="e.g. Prayagraj")

    if st.button("🔍 Search Hospitals", type="primary", use_container_width=True):
        if not city:
            st.warning("Please enter a city name.")
        else:
            try:
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="medisense_ai_app")
                location = geolocator.geocode(f"{city}, India")

                if location is None:
                    st.warning("Couldn't find that city. Try a bigger nearby city name.")
                else:
                    with st.spinner("Searching nearby hospitals..."):
                        hospitals, search_error = find_nearby_hospitals(location.latitude, location.longitude)
                    st.session_state.hospitals = hospitals
                    st.session_state.map_center = (location.latitude, location.longitude)

                    if hospitals:
                        st.success(f"Found {len(hospitals)} hospitals near {city}.")
                    elif search_error:
                        st.error(
                            f"Hospital search failed — this is a network/connectivity issue, not your input. "
                            f"Details: {search_error}"
                        )
                        st.info(
                            "Try again in a moment — public OpenStreetMap servers occasionally rate-limit or time out. "
                            "If this keeps happening, check that this machine has normal internet access."
                        )
                    else:
                        st.info(
                            f"The search worked, but no hospitals are tagged in OpenStreetMap within 8 km of {city}. "
                            "Try a bigger nearby city name."
                        )
            except ImportError:
                st.error("Please install: `pip install geopy`")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.hospitals:
        try:
            import folium
            from streamlit_folium import st_folium

            lat, lng = st.session_state.map_center
            m = folium.Map(location=[lat, lng], zoom_start=13)
            folium.Marker([lat, lng], popup="You are here", icon=folium.Icon(color="blue")).add_to(m)
            for h in st.session_state.hospitals[:20]:
                folium.Marker([h["lat"], h["lng"]], popup=h["name"],
                               icon=folium.Icon(color="red", icon="plus-sign")).add_to(m)
            st_folium(m, width=700, height=350)
        except ImportError:
            st.info("Install `streamlit-folium` to see the map: `pip install streamlit-folium`")

        st.divider()
        st.markdown("#### 📅 Book an Appointment (Demo)")

        hospital_names = [h["name"] for h in st.session_state.hospitals]

        with st.form("booking_form"):
            patient_name = st.text_input("Your Name")
            email = st.text_input("Email")
            hospital_choice = st.selectbox("Select Hospital", hospital_names)
            preferred_date = st.date_input("Preferred Date")
            submitted = st.form_submit_button("📅 Book Appointment", use_container_width=True)

            if submitted:
                if not patient_name or not email:
                    st.warning("Please fill in your name and email.")
                else:
                    import sqlite3
                    disease = st.session_state.prediction[0] if st.session_state.prediction else "N/A"
                    conn = sqlite3.connect("appointments.db")
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS appointments (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            patient_name TEXT, email TEXT, hospital_name TEXT,
                            predicted_disease TEXT, preferred_date TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    cursor.execute(
                        "INSERT INTO appointments (patient_name, email, hospital_name, predicted_disease, preferred_date) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (patient_name, email, hospital_choice, disease, str(preferred_date))
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"✅ Appointment booked at **{hospital_choice}** on **{preferred_date}**. (Demo confirmation)")

    st.write("")
    if st.button("🏠 Start Over"):
        go_home()
        st.rerun()

st.divider()
st.caption("MediSense AI — Symptom & Skin Checker · Urgency Scoring · Hospital Locator")
