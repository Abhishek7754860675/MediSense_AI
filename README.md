# MediSense AI — AI Health Diagnosis Assistant

## Files
- `MediSense_AI_Complete.ipynb` — run this in **Google Colab (GPU on)** first. It trains/builds all 4 modules and saves the files the app needs:
  - `symptom_disease_model.pkl`, `label_encoder.pkl`, `symptom_columns.pkl` (Module 1)
  - `skin_disease_model.keras`, `skin_class_indices.pkl` (Module 2)
  - `severity_map.pkl` (Module 3)
  - Module 4 (hospital search) needs no saved file — it works live
- `medisense_app.py` — the Streamlit app
- `PROJECT_SUMMARY.md` — full project notes (what's done, how each part works)

## How to run

1. Open `MediSense_AI_Complete.ipynb` in Google Colab (Runtime → Change runtime type → GPU)
2. Run every cell top to bottom. When asked, upload `Training.csv` / `Testing.csv` (Module 1) and your Kaggle `kaggle.json` (Module 2, **Legacy API Key** format)
3. Once it finishes, download all the generated `.pkl` / `.keras` files from the Colab Files sidebar
4. Put those files in the **same folder** as `medisense_app.py`
5. Install requirements:
   ```
   pip install streamlit tensorflow pillow joblib pandas numpy scikit-learn geopy folium streamlit-folium requests
   ```
6. Run the app:
   ```
   streamlit run medisense_app.py
   ```

## App Flow
1. Home page — choose **Symptom Checker** or **Skin Checker**
2. Enter symptoms / upload an image → get predicted condition + confidence + urgency level + a short info note
3. Click **"Locate Best Hospital & Book Appointment"** → enter your city → see nearby hospitals on a map → pick one and submit a demo booking
