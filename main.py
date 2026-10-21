import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from Model.model import (
    DEFAULT_DATASET,
    TARGET,
    load_dataset,
    train_and_evaluate,
    make_message_features,
    predict_message,
    save_model,
)

st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Message Intelligence System")
st.caption("Classify messages as legitimate or spam using KNN, SVM, and Naive Bayes.")

with st.sidebar:
    st.header("📂 Dataset")
    uploaded_file = st.file_uploader(
        "Upload dataset (Excel or CSV)",
        type=["xlsx", "xls", "csv"],
        help="Expected target column: spam_label (0 = Legitimate, 1 = Spam).",
    )
    st.info("If no file is uploaded, the app looks for the project Excel file next to main.py.")

@st.cache_data(show_spinner=False)
def read_uploaded_data(file_bytes, filename):
    return load_dataset(io.BytesIO(file_bytes) if filename.lower().endswith(".csv") else io.BytesIO(file_bytes))

try:
    if uploaded_file is not None:
        # Use a named buffer so load_dataset can detect CSV vs Excel.
        class NamedBytesIO(io.BytesIO):
            pass
        file_buffer = NamedBytesIO(uploaded_file.getvalue())
        file_buffer.name = uploaded_file.name
        data = load_dataset(file_buffer)
        dataset_label = uploaded_file.name
    elif DEFAULT_DATASET.exists():
        data = load_dataset(DEFAULT_DATASET)
        dataset_label = DEFAULT_DATASET.name
    else:
        st.warning("Upload your Excel/CSV dataset to start, or place Message_Intelligence_Dataset_5200.xlsx next to main.py.")
        st.stop()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if "training_results" not in st.session_state:
    st.session_state.training_results = None

st.success(f"Loaded **{dataset_label}** — {len(data):,} rows")
overview_tab, train_tab, predict_tab, data_tab = st.tabs([
    "📊 Overview", "🧠 Model Comparison", "🔎 Predict a Message", "🗃️ Dataset"
])

with overview_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Messages", f"{len(data):,}")
    c2.metric("Spam Messages", f"{int((data[TARGET] == 1).sum()):,}")
    c3.metric("Legitimate Messages", f"{int((data[TARGET] == 0).sum()):,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Label Distribution")
        counts = data[TARGET].map({0: "Legitimate", 1: "Spam"}).value_counts()
        fig, ax = plt.subplots()
        counts.reindex(["Legitimate", "Spam"]).plot(kind="bar", ax=ax)
        ax.set_xlabel("Message class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Dataset Health")
        health = pd.DataFrame({
            "Column": data.columns,
            "Missing values": [int(data[c].isna().sum()) for c in data.columns],
        })
        st.dataframe(health, use_container_width=True, hide_index=True)

    st.markdown(
        "**Target meaning:** `0` = Legitimate Message · `1` = Spam Message. "
        "Model metrics are calculated on a stratified held-out test set."
    )

with train_tab:
    st.subheader("Train and compare classifiers")
    test_fraction = st.slider("Test-set percentage", min_value=10, max_value=40, value=20, step=5)
    if st.button("🚀 Train models", type="primary"):
        with st.spinner("Training KNN, SVM, and Naive Bayes..."):
            try:
                models, metrics, details = train_and_evaluate(
                    data, test_size=test_fraction / 100
                )
                st.session_state.training_results = {
                    "models": models,
                    "metrics": metrics,
                    "details": details,
                }
                # Save the top F1-score model and all individual models.
                for name, fitted_model in models.items():
                    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    save_model(fitted_model, Path("saved_models") / f"{safe_name}.joblib")
                best_name = metrics.iloc[0]["Model"]
                save_model(models[best_name], Path("saved_models") / "best_model.joblib")
                st.success(f"Training complete. Best F1 score on this test split: {best_name}")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    trained = st.session_state.training_results
    if trained:
        metrics = trained["metrics"]
        details = trained["details"]
        st.subheader("Evaluation Metrics")
        st.dataframe(
            metrics.style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                "Recall": "{:.3f}", "F1 Score": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Metric Comparison")
        chart_data = metrics.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
        st.bar_chart(chart_data)

        selected_model = st.selectbox("View confusion matrix for", list(trained["models"].keys()))
        matrix = details["confusion_matrices"][selected_model]
        fig, ax = plt.subplots()
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Legitimate", "Spam"],
            yticklabels=["Legitimate", "Spam"],
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_title(f"{selected_model} — Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification Report")
        report = pd.DataFrame(details["reports"][selected_model]).transpose()
        st.dataframe(report.round(3), use_container_width=True)

        st.download_button(
            "⬇️ Download metrics as CSV",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Train models** to generate evaluation metrics and plots.")

with predict_tab:
    st.subheader("Test a message")
    st.write("Enter a message and sender-behavior estimates. Predictions are available after training.")
    with st.form("prediction_form"):
        message = st.text_area(
            "Message text",
            placeholder="Example: Congratulations! Claim your free prize now...",
            height=130,
        )
        a, b, c = st.columns(3)
        with a:
            activity = st.number_input("Sender activity score", min_value=0.0, value=50.0)
        with b:
            account_age = st.number_input("Sender account age (days)", min_value=0.0, value=180.0)
        with c:
            messages_24h = st.number_input("Messages sent in last 24 hours", min_value=0.0, value=5.0)
        model_choice = st.selectbox("Classifier", ["SVM (RBF)", "KNN", "Naive Bayes"])
        submitted = st.form_submit_button("Classify message", type="primary")

    if submitted:
        trained = st.session_state.training_results
        if not message.strip():
            st.warning("Please enter a message.")
        elif not trained:
            st.warning("Train the models first in the Model Comparison tab.")
        else:
            features = make_message_features(
                message,
                sender_activity_score=activity,
                sender_account_age_days=account_age,
                messages_sent_last_24h=messages_24h,
            )
            label, probabilities = predict_message(trained["models"][model_choice], features)
            if label == 1:
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: LEGITIMATE")
            if probabilities:
                st.write("Model probability estimates")
                st.progress(probabilities.get(1, 0.0), text=f"Spam probability: {probabilities.get(1, 0.0):.1%}")
                st.caption("Probability estimates depend on the selected classifier and dataset.")
            st.caption("Note: keyword scores for typed messages are estimated using a simple keyword list; match the dataset's original feature-engineering logic for production use.")

with data_tab:
    st.subheader("Dataset preview")
    st.dataframe(data.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download loaded dataset as CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="message_intelligence_dataset.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Educational project · Always review false positives and false negatives before real-world deployment.")
