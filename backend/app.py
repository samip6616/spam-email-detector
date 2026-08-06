import os
import pickle
import re
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------------------------
# 1. Path Management & Automatic Folder Creation
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DB_DIR = BASE_DIR.parent / "database"
MODEL_DIR = BASE_DIR.parent / "model"

# Create required directories automatically
for folder in [DATA_DIR, DB_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "spam.csv"
DB_FILE = DB_DIR / "spam_classifier.db"
MODEL_FILE = MODEL_DIR / "model.pkl"
VECTORIZER_FILE = MODEL_DIR / "vectorizer.pkl"


# -------------------------------------------------------------------
# 2. Text Preprocessing & Security Analysis
# -------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Cleans and normalizes email body text for model ingestion."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_urls(text: str) -> list:
    """Detects embedded URLs or domain references in the email."""
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.\S+"
    )
    return url_pattern.findall(text)


# -------------------------------------------------------------------
# 3. CSV Dataset Loader & Normalizer
# -------------------------------------------------------------------
def load_csv_dataset(file_path: Path) -> pd.DataFrame:
    """Loads and standardizes baseline dataset from spam.csv."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Please place 'spam.csv' in the target directory."
        )

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    # Map headers dynamically
    col_map = {str(col).lower().strip(): col for col in df.columns}
    text_col = next(
        (
            col_map[c]
            for c in ["text", "v2", "message", "email", "body", "content"]
            if c in col_map
        ),
        None,
    )
    label_col = next(
        (
            col_map[c]
            for c in ["label", "v1", "category", "target", "class", "type"]
            if c in col_map
        ),
        None,
    )

    if not text_col or not label_col:
        if len(df.columns) >= 2:
            label_col, text_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(
                "CSV must contain at least two columns for label and text."
            )

    cleaned_df = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[label_col].astype(str).str.strip().str.capitalize(),
        }
    )

    cleaned_df.dropna(subset=["text", "label"], inplace=True)
    cleaned_df["label"] = cleaned_df["label"].replace(
        {"1": "Spam", "0": "Ham", "Pos": "Spam", "Neg": "Ham"}
    )
    cleaned_df["processed_text"] = cleaned_df["text"].apply(preprocess_text)
    return cleaned_df


# -------------------------------------------------------------------
# 4. Database Initialization & Persistence
# -------------------------------------------------------------------
def init_db():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    email_text TEXT,
                    prediction TEXT,
                    confidence REAL,
                    user_feedback TEXT
                )
            """
            )
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database Initialization Error: {e}")


init_db()


def log_prediction(text: str, prediction: str, confidence: float) -> int:
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                INSERT INTO history (timestamp, email_text, prediction, confidence, user_feedback)
                VALUES (?, ?, ?, ?, ?)
            """,
                (now, text, prediction, confidence, "Unverified"),
            )
            conn.commit()
            return cursor.lastrowid
    except sqlite3.Error as e:
        st.error(f"Failed to log prediction to database: {e}")
        return -1


def update_feedback(pred_id: int, feedback_label: str):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE history SET user_feedback = ? WHERE id = ?",
                (feedback_label, pred_id),
            )
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Failed to record feedback: {e}")


def fetch_history() -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_FILE) as conn:
            return pd.read_sql_query(
                "SELECT * FROM history ORDER BY id DESC", conn
            )
    except Exception:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "email_text",
                "prediction",
                "confidence",
                "user_feedback",
            ]
        )


# -------------------------------------------------------------------
# 5. Cached Model Training & Persistence
# -------------------------------------------------------------------
def save_model(model: MultinomialNB, vectorizer: TfidfVectorizer):
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
        with open(VECTORIZER_FILE, "wb") as f:
            pickle.dump(vectorizer, f)
    except Exception as e:
        st.error(f"Error saving model artifacts: {e}")


@st.cache_resource
def load_or_train_model():
    """Loads model from disk or trains baseline from spam.csv with caching."""
    if MODEL_FILE.exists() and VECTORIZER_FILE.exists():
        try:
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)
            with open(VECTORIZER_FILE, "rb") as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        except Exception:
            st.warning("Saved model corrupted. Retraining from baseline CSV...")

    # Baseline training from CSV
    try:
        baseline_df = load_csv_dataset(CSV_FILE)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(baseline_df["processed_text"])
        model = MultinomialNB()
        model.fit(X, baseline_df["label"])
        save_model(model, vectorizer)
        return model, vectorizer
    except Exception as e:
        st.error(f"Initialization Failed: {e}")
        return None, None


def retrain_from_feedback():
    """Combines spam.csv baseline with verified database feedback to retrain."""
    try:
        baseline_df = load_csv_dataset(CSV_FILE)

        with sqlite3.connect(DB_FILE) as conn:
            feedback_df = pd.read_sql_query(
                "SELECT email_text, user_feedback FROM history WHERE user_feedback IN ('Spam', 'Ham')",
                conn,
            )

        if not feedback_df.empty:
            feedback_df = feedback_df.rename(
                columns={"email_text": "text", "user_feedback": "label"}
            )
            feedback_df["processed_text"] = feedback_df["text"].apply(
                preprocess_text
            )
            full_df = pd.concat(
                [
                    baseline_df[["processed_text", "label"]],
                    feedback_df[["processed_text", "label"]],
                ],
                ignore_index=True,
            )
        else:
            full_df = baseline_df

        new_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = new_vectorizer.fit_transform(full_df["processed_text"])
        new_model = MultinomialNB()
        new_model.fit(X, full_df["label"])

        save_model(new_model, new_vectorizer)
        st.cache_resource.clear()  # Evict cached model instance
        return len(baseline_df), len(feedback_df)
    except Exception as e:
        st.error(f"Retraining failed: {e}")
        return 0, 0


# Load model state
model, vectorizer = load_or_train_model()

# -------------------------------------------------------------------
# 6. Streamlit User Interface
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Classifier", page_icon="📧", layout="wide"
)

st.title("📧 Email Spam Classifier & Intelligence Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["🔍 Analyze Email", "📊 History & Dashboard", "⚙️ Model Management"]
)

# -------------------------------------------------------------------
# TAB 1: CLASSIFIER & SECURITY DETECTION
# -------------------------------------------------------------------
with tab1:
    st.subheader("Analyze Suspicious Message Body")
    email_input = st.text_area(
        "Paste raw email body here:",
        height=180,
        placeholder="e.g. Urgent! Claim your free gift card now by visiting http://example-phish.com",
    )

    # Real-time URL Threat Detection
    detected_urls = detect_urls(email_input)
    if detected_urls:
        st.warning(
            f"⚠️ **URL Security Warning:** Detected {len(detected_urls)} embedded link(s) in input text:\n"
            + "\n".join([f"- `{u}`" for u in detected_urls])
        )

    if st.button("Classify Email", type="primary", use_container_width=True):
        if not email_input.strip():
            st.warning("Please paste or type text content before analyzing.")
        elif model is None or vectorizer is None:
            st.error("Classifier model is not ready. Check data files.")
        else:
            processed = preprocess_text(email_input)
            X_input = vectorizer.transform([processed])
            prediction = str(model.predict(X_input)[0])
            probabilities = model.predict_proba(X_input)[0]
            classes = list(model.classes_)

            spam_prob = float(probabilities[classes.index("Spam")]) * 100
            ham_prob = float(probabilities[classes.index("Ham")]) * 100
            confidence = max(spam_prob, ham_prob)

            pred_id = log_prediction(
                email_input, prediction, round(confidence, 2)
            )
            st.session_state["current_pred_id"] = pred_id
            st.session_state["last_prediction"] = prediction

            st.divider()

            col_res, col_prob = st.columns([1, 1])

            with col_res:
                if prediction == "Spam":
                    st.error(f"### Classification Result: 🚨 **SPAM**")
                else:
                    st.success(f"### Classification Result: ✅ **HAM (Safe)**")
                st.caption(f"Log Sequence Reference: #{pred_id}")

            with col_prob:
                st.markdown("#### Class Probabilities")
                st.write(f"**Spam:** {spam_prob:.2f}%")
                st.progress(spam_prob / 100.0)
                st.write(f"**Ham:** {ham_prob:.2f}%")
                st.progress(ham_prob / 100.0)

    # Active Learning Feedback Component
    if "current_pred_id" in st.session_state:
        st.write("---")
        st.subheader("Help Improve Model Accuracy")
        st.caption("Was this prediction correct?")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            if st.button("👍 Correct Prediction", use_container_width=True):
                update_feedback(
                    st.session_state["current_pred_id"],
                    st.session_state["last_prediction"],
                )
                st.toast("Verified feedback recorded!", icon="✅")

        with f_col2:
            if st.button("🚨 Incorrect (Should be Spam)", use_container_width=True):
                update_feedback(st.session_state["current_pred_id"], "Spam")
                st.toast("Corrected label saved: Spam", icon="🚨")

        with f_col3:
            if st.button("✅ Incorrect (Should be Ham)", use_container_width=True):
                update_feedback(st.session_state["current_pred_id"], "Ham")
                st.toast("Corrected label saved: Ham", icon="✅")


# -------------------------------------------------------------------
# TAB 2: DASHBOARD METRICS, HISTORY & EXPORT
# -------------------------------------------------------------------
with tab2:
    history_df = fetch_history()

    # KPI Summary Dashboard
    st.subheader("📈 System Metrics & Analytics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_preds = len(history_df)
    spam_preds = (
        len(history_df[history_df["prediction"] == "Spam"])
        if total_preds > 0
        else 0
    )
    ham_preds = (
        len(history_df[history_df["prediction"] == "Ham"])
        if total_preds > 0
        else 0
    )
    verified_count = (
        len(history_df[history_df["user_feedback"].isin(["Spam", "Ham"])])
        if total_preds > 0
        else 0
    )

    kpi1.metric("Total Predictions", total_preds)
    kpi2.metric("Spam Identified", spam_preds)
    kpi3.metric("Ham Identified", ham_preds)
    kpi4.metric("Feedback Samples", verified_count)

    st.divider()

    # Search & Filter Controls
    st.subheader("📜 Historical Audit Log")
    col_search, col_filter, col_export = st.columns([2, 1, 1])

    with col_search:
        search_query = st.text_input(
            "🔎 Search logs by email body keyword:", placeholder="Type to filter..."
        )

    with col_filter:
        label_filter = st.selectbox(
            "Filter Prediction:", ["All", "Spam", "Ham"]
        )

    # Apply filters
    filtered_df = history_df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df["email_text"].str.contains(
                search_query, case=False, na=False
            )
        ]
    if label_filter != "All":
        filtered_df = filtered_df[filtered_df["prediction"] == label_filter]

    with col_export:
        st.write(" ")
        st.write(" ")
        if not filtered_df.empty:
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export History (CSV)",
                data=csv_data,
                file_name=f"spam_classifier_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if filtered_df.empty:
        st.info("No prediction logs match your criteria.")
    else:
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": "ID",
                "timestamp": "Timestamp",
                "email_text": "Email Body",
                "prediction": "Prediction",
                "confidence": st.column_config.NumberColumn(
                    "Confidence", format="%.2f %%"
                ),
                "user_feedback": "User Feedback",
            },
        )


# -------------------------------------------------------------------
# TAB 3: ACTIVE LEARNING RETRAINING LOOP
# -------------------------------------------------------------------
with tab3:
    st.subheader("⚙️ Active Learning Model Management")
    st.write(
        "Retrain the Naive Bayes classifier by aggregating the baseline `spam.csv` "
        "dataset with newly verified human feedback logged in the database."
    )

    with sqlite3.connect(DB_FILE) as conn:
        fb_count = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM history WHERE user_feedback IN ('Spam', 'Ham')",
            conn,
        ).iloc[0]["cnt"]

    st.info(f"💡 Currently **{fb_count} verified feedback records** available for retraining.")

    if st.button("🚀 Retrain Model Now", type="primary"):
        with st.spinner("Processing dataset and updating model weights..."):
            base_n, fb_n = retrain_from_feedback()
            if base_n > 0:
                st.success(
                    f"Model successfully retrained on **{base_n + fb_n} total samples** "
                    f"({base_n} baseline + {fb_n} verified feedback)."
                )