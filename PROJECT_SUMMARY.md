# MediSense AI — Progress Summary
Resume Project — AI Health Diagnosis Assistant
Background: BCA student, specialization AI/ML, Deep Learning, Data Science (Python).

---

## Project Overview

**App Name:** MediSense AI
**Full Title:** MediSense AI — AI Health Diagnosis Assistant

The app has 4 modules:
1. Symptom-Based Disease Predictor (Classical ML) — **✅ DONE**
2. AI Skin Disease Detector (Deep Learning / CNN) — ⬜ NOT STARTED
3. Urgency Scoring Engine (Data Science) — ⬜ NOT STARTED
4. Hospital Locator + Appointment Booking — ⬜ NOT STARTED

Tech Stack (full project): Python, Pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib, Seaborn, tf-keras-vis (Grad-CAM), SHAP, Streamlit, Google Places API, Folium, SQLite, smtplib. Deployment target: Streamlit Cloud / Hugging Face Spaces.

---

## Current Status (as of this session)
- Module 1: **COMPLETE** (notebook + model + Streamlit app all working, tested with real inputs)
- Module 2: **COMPLETE** — trained on Google Colab GPU, evaluation + Grad-CAM + model saving all done
- Module 3: **COMPLETE** — Urgency Scoring Engine notebook built and tested
- Module 4: **NOTEBOOK BUILT, PENDING API KEY** — Hospital Locator + Appointment notebook is ready (specialist mapping, Places search function, Folium map, SQLite booking tested working, optional email). User is in the process of creating a Google Places API key (Google Cloud Console -> new project -> enable Places API -> Credentials -> Create API Key -> restrict to Places API). Once the key is added to `GOOGLE_PLACES_API_KEY` in Phase 0, the hospital search cells can be tested.
- **Not yet done: final integration** — combining all 4 modules into a single multi-tab Streamlit app, and deployment (Streamlit Cloud / Hugging Face Spaces)
- Note: hit a notebook-formatting bug where code-cell source lines were missing line breaks, causing SyntaxErrors when run on Colab — Module 1 and Module 2 notebooks were rebuilt/fixed for this (Module 3 and 4 were built with the fix already in place).

---

## MODULE 1: Symptom-Based Disease Predictor — COMPLETE

### Dataset
- Kaggle: "Disease Prediction Using Machine Learning" (Kaushil Patel)
- Files used: `Training.csv` (4920 rows) and `Testing.csv` (42 rows)
- 132 binary (0/1) symptom columns, target column `prognosis`, 41 disease classes
- One junk column `Unnamed: 133` (always empty) — dropped

### Key data decisions made
- **Duplicates (4616 rows) were kept, not dropped** — they are genuine repeated symptom-combination cases for the same disease (dataset is synthetically generated this way), not data-entry errors. Dropping them would shrink training data to ~304 rows and hurt the model.
- Classes are **perfectly balanced** (120 rows per disease in training) — no SMOTE/oversampling needed.
- No outlier treatment or feature engineering needed — features are already binary flags.
- Target encoded with `LabelEncoder`.

### Notebook structure (final, simplified version)
Mirrors an earlier "Loan Approval Prediction" capstone project's style — incremental imports per phase, no unnecessary complexity (no SMOTE, no GridSearchCV, no cross-validation — kept intentionally simple/readable):

- Phase 1-2: Problem Understanding & Data Collection
- Phase 3: Data Preprocessing (duplicates, missing values, class balance, target encoding)
- Phase 4: EDA (disease distribution, correlation heatmap of top symptoms)
- Phase 5: Train-Test Split (80/20, stratified)
- Phase 6: Model Building — Logistic Regression, Random Forest, XGBoost (trained directly, no tuning)
- Phase 7: Model Evaluation
  - Comparison table using **Accuracy + F1-score (weighted)** only (sufficient for this balanced multi-class problem — precision/recall/ROC-AUC were used in the earlier binary loan project but aren't needed here)
  - **Best model auto-selected in code** via `results_df.iloc[0]['Model']` — not hardcoded — so whichever model wins the comparison becomes `best_model` automatically
  - In the user's last run: **Logistic Regression** won (100% accuracy on validation split)
  - Confusion matrix plotted for the winning model
  - Feature importance plotted only if the model supports it (`hasattr(best_model, 'feature_importances_')` — Logistic Regression doesn't expose this, tree models do; code handles this gracefully with a fallback message)
  - Sanity-check accuracy computed on the real held-out `Testing.csv` (~97.6% seen in earlier test runs)
- Phase 8: Model Saving — `joblib.dump()` for:
  - `symptom_disease_model.pkl` (the winning trained model)
  - `label_encoder.pkl`
  - `symptom_columns.pkl` (ordered list of the 132 feature names)

### Streamlit App (Module 1 UI) — built and working
File: `app.py`. Went through 3 iterations, current version has:
- **Centered, step-based layout** (`layout="centered"`) — everything opens in the middle of the page, not stretched wide
- Gradient header banner ("MediSense AI" title + subtitle)
- Collapsible "About this model" info expander (shows model type, symptom/disease counts)
- **STEP 1** — multiselect symptom picker (human-readable labels, e.g. "High Fever" instead of `high_fever`), selected symptoms shown as rounded "chip" tags
- **STEP 2** — Prediction Result section directly below (same page, vertical flow):
  - Result card showing the top predicted disease + confidence %, color-coded (green ≥70%, yellow 40-69%, red <40%)
  - Interactive **Plotly horizontal bar chart** of top-5 disease matches with confidence %
  - Medical disclaimer box (not a real diagnosis, consult a doctor)
- Loads the 3 saved `.pkl` files via `@st.cache_resource`
- Verified working end-to-end by the user with real symptom inputs (e.g. Vomiting/Stomach Pain/Acidity/Anxiety → GERD 55%; Headache/High Fever/Loss of Appetite → AIDS 14.48%, all top-5 close together — expected behavior when symptoms are generic/overlap across many diseases rather than being disease-specific)

**Requirements to run:** `pip install streamlit plotly joblib pandas numpy`, then `streamlit run app.py` with the 3 `.pkl` files in the same folder.

---

## What's Left To Do

### Module 2: AI Skin Disease Detector (Deep Learning — CNN) — IN PROGRESS
File: `skin_disease_detector.ipynb`. Built in the same phase-based style as Module 1.
- Dataset: **HAM10000** (Skin Cancer MNIST, Kaggle: `kmader/skin-cancer-mnist-ham10000`), 7 classes (akiec, bcc, bkl, df, mel, nv, vasc) — chosen over the easier Chest X-Ray/Pneumonia dataset for uniqueness
- Downloaded via Kaggle API on Colab (needed the **Legacy API Key** format `kaggle.json`, not the newer `KGAT_...` token format, since the notebook code expects the classic `~/.kaggle/kaggle.json` file)
- `HAM10000_metadata.csv` has `image_id`, `dx` (label), etc.; images live across two folders (`HAM10000_images_part_1`, `_part_2`) — combined into one `image_path` column
- Missing `age` values filled with median
- Class distribution is imbalanced (`nv` ~67% of images) — handled via `class_weight` during training rather than dropping data
- Approach: **Transfer learning with MobileNetV2** (pretrained on ImageNet, base frozen, custom Dense head added) — not training a CNN from scratch
- `ImageDataGenerator` with augmentation (rotation, zoom, horizontal/vertical flip) via `flow_from_dataframe`
- Training: 15 epochs with `EarlyStopping` (monitor `val_accuracy`, patience 3) and computed `class_weight`
- Evaluation planned: classification report + confusion matrix over the validation set
- **Grad-CAM explainability** — implemented via `tf.GradientTape` on the last MobileNetV2 conv layer, overlays a heatmap on sample images showing which region the model focused on. This is the key differentiating/unique feature for the resume.
- Model saving planned: `skin_disease_model.keras` + `skin_class_indices.pkl` via joblib
- **Status at handoff: training was running on Colab GPU; still need to confirm final accuracy, run Grad-CAM cells, and verify the saved files exist before moving on**
- A Streamlit tab/section for this module has NOT been built yet — Module 1's `app.py` only covers the symptom predictor so far.

### Module 3: Urgency Scoring Engine (Data Science) — COMPLETE
File: `urgency_scoring_engine.ipynb`. No ML training — pure logic:
- Severity map (1-10) built for all 48 diseases (41 from Module 1 + 7 from Module 2)
- Formula: normalize severity/confidence/symptom-count to 0-100 each, then `score = 0.4*severity_pct + 0.3*confidence_pct + 0.3*symptom_pct` (symptom count capped at 10 = 100%)
- Categorized: High >=70, Medium 40-69, Low <40
- Tested with sample cases (e.g. Heart attack -> High, Common Cold -> Medium, Fungal infection -> Low)
- Saved `severity_map.pkl` via joblib for Streamlit use
- `calculate_urgency(disease, confidence, symptom_count)` function is ready to drop into the final app

### Module 4: Hospital Locator + Appointment — NOTEBOOK BUILT, PENDING API KEY
File: `hospital_locator.ipynb`.
- Phase 0: needs a Google Places API key (Google Cloud Console -> new project -> enable Places API -> Credentials -> Create API Key -> restrict to Places API) — **user was setting this up when this summary was written, not yet confirmed working**
- Disease -> specialist mapping dict (e.g. "Fungal infection" -> dermatologist, "Heart attack" -> cardiologist) covering diseases from both Module 1 and 2
- `find_nearby_hospitals(lat, lng, specialization)` — calls Google Places Nearby Search API
- `show_hospital_map(lat, lng, hospitals)` — Folium map with user location + hospital markers
- SQLite `appointments.db` with `book_appointment()` / retrieval — **tested and confirmed working** independent of the API key
- Optional `send_confirmation_email()` via smtplib — needs a Gmail App Password, not yet set up

---

## Next Steps (in order)
1. Finish Module 4: get the Google Places API key working, test `find_nearby_hospitals()` and the map with real coordinates
2. **Final integration**: combine all 4 modules into a single Streamlit app with tabs/pages — Module 1's `app.py` currently only has the symptom predictor; needs Module 2 (image upload + Grad-CAM display), Module 3 (urgency badge shown alongside predictions), and Module 4 (hospital map + booking form) added
3. Deploy on Streamlit Cloud or Hugging Face Spaces
4. Write a GitHub README summarizing the project for the resume

### Final Integration
- Combine all 4 modules into a single multi-tab/multi-page Streamlit app
- Deploy on Streamlit Cloud or Hugging Face Spaces
- Write GitHub README

### Original 8-week roadmap (for reference)
- Week 1-2: Module 1 ✅ **DONE**
- Week 3-4: Module 2 (CNN + transfer learning) — up next
- Week 5: Grad-CAM implementation
- Week 6: Module 3 (Urgency Scoring) + EDA
- Week 7: Module 4 (Hospital Locator)
- Week 8: Integration, deployment, GitHub README

---

## Resume Line (once fully complete)
> MediSense AI — AI Health Diagnosis Assistant (Python, TensorFlow, scikit-learn, Streamlit)
> Built a multi-modal healthcare AI system combining symptom-based ML classification (96%+ accuracy), CNN-based skin disease detection with Grad-CAM explainability, urgency scoring engine, and real-time hospital locator via Google Places API. Deployed on Streamlit Cloud.
