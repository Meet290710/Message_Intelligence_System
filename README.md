::: {align="center"}
# 🛡️ Message Intelligence System

### A machine-learning project for classifying messages as **Legitimate** or **Spam**

**Turn message data into useful insights.** Explore a dataset, compare
classification models, review evaluation metrics, and test a message in
an interactive Streamlit dashboard.
:::

------------------------------------------------------------------------

## ✨ Project at a Glance

The **Message Intelligence System** is an interactive machine-learning
application designed to identify whether a message is likely to be spam
or legitimate. It combines message text with numerical signals---such as
URL count, message length, keyword scores, and sender activity---to
train and compare three classifiers.

  -----------------------------------------------------------------------
                                      Details
  ----------------------------------- -----------------------------------
  🎯 **Goal**                         Classify messages as legitimate or
                                      spam

  🧠 **Models**                       K-Nearest Neighbors (KNN), Support
                                      Vector Machine (SVM), and Naive
                                      Bayes

  🖥️ **Interface**                    Streamlit web application

  📊 **Evaluation**                   Accuracy, precision, recall, F1
                                      score, classification report, and
                                      confusion matrix

  📁 **Data input**                   Excel or CSV dataset

  🧪 **Use case**                     Learning, experimentation, and
                                      model comparison
  -----------------------------------------------------------------------

> \[!NOTE\] This is an educational project. Predictions depend on the
> training data and feature-engineering choices; they should not be
> treated as a guarantee that a message is safe or malicious.

## 🚀 Features

-   **📊 Dataset overview** --- inspect message counts, label
    distribution, and missing values.
-   **🧠 Compare three models** --- train KNN, SVM (RBF), and
    Multinomial Naive Bayes on the same train/test split.
-   **📈 Understand model performance** --- compare accuracy, precision,
    recall, and F1 score.
-   **🧩 Inspect a confusion matrix** --- see correct and incorrect
    predictions for the selected model.
-   **🔎 Try a message** --- enter text and sender-behavior estimates to
    get a predicted class and, where available, a model probability
    estimate.
-   **📤 Upload your own data** --- load a compatible Excel or CSV
    dataset.
-   **⬇️ Export results** --- download the evaluation metrics or loaded
    dataset as CSV.
-   **💾 Save trained models** --- model pipelines can be saved as
    `.joblib` files.

## 🚀 Live Demo

<p align="center">
  <a href="https://message-intelligence-system1.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Launch-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch Live App">
  </a>
</p>

Try the **Message Intelligence System** directly in your browser — no installation required!

🔗 **Live Application:** [Open Message Intelligence System](https://message-intelligence-system1.streamlit.app/)

## 🧰 Tech Stack

  ----------------------------------------------------------------------------------
  Technology                                     Purpose
  ---------------------------------------------- -----------------------------------
  [Python](https://www.python.org/)              Core programming language

  [Streamlit](https://streamlit.io/)             Interactive web dashboard

  [pandas](https://pandas.pydata.org/) and       Data loading and numerical
  [NumPy](https://numpy.org/)                    processing

  [scikit-learn](https://scikit-learn.org/)      Text preprocessing, model training,
                                                 and evaluation

  [Matplotlib](https://matplotlib.org/) and      Visualizations
  [Seaborn](https://seaborn.pydata.org/)         

  [openpyxl](https://openpyxl.readthedocs.io/)   Excel file support

  [joblib](https://joblib.readthedocs.io/)       Saving trained model pipelines
  ----------------------------------------------------------------------------------

## 🗂️ Repository Structure

``` text
pr_4/
├── main.py
├── requirements.txt
├── .gitignore.txt
├── data/
│   └── Message_Intelligence_Dataset_5200.xlsx
├── Model/
│   └── model.py
├── README.md   
└── Message_Intelligence_System_Meet_mehta_12237.mp4
```


## ⚙️ Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the URL and repository folder with your own GitHub repository
details.

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check the dataset path

The current `Model/model.py` expects the default Excel file at:

``` text
Model/Message_Intelligence_Dataset_5200.xlsx
```

However, the supplied repository structure places the workbook in
`data/`. Choose one of these options:

-   **Option A:** Move or copy the workbook into `Model/`.
-   **Option B (recommended):** Update `DEFAULT_DATASET` in
    `Model/model.py` to point to the `data/` folder:

``` python
DEFAULT_DATASET = BASE_DIR.parent / "data" / "Message_Intelligence_Dataset_5200.xlsx"
```

This path assumes the `Model` directory is directly inside the
repository root, as shown above.

### 5. Launch the app

From the repository root, run:

``` bash
streamlit run main.py
```

Streamlit will print a local URL---usually `http://localhost:8501`. Open
it in your browser.

## 🧭 How to Use the Dashboard

1.  **Load data:** use the included dataset or upload a compatible
    `.xlsx`, `.xls`, or `.csv` file in the sidebar.
2.  **Explore the overview:** review the total number of messages,
    spam/legitimate counts, label distribution, and missing values.
3.  **Train models:** open **Model Comparison**, choose the test-set
    percentage, and click **Train models**.
4.  **Compare metrics:** review the score table and metric chart.
    Results are calculated on a stratified held-out test set.
5.  **Inspect errors:** select a classifier to view its confusion matrix
    and classification report.
6.  **Classify a message:** open **Predict a Message**, enter the
    message and sender estimates, select a classifier, and submit.
7.  **Export data:** download the metrics or dataset from the relevant
    tabs.

## 🧠 Machine-Learning Workflow

``` text
Dataset (Excel / CSV)
        │
        ▼
Validate required columns and labels
        │
        ▼
Text features ──► TF-IDF (unigrams + bigrams)
Numeric features ► Median imputation + Min-Max scaling
        │
        ▼
Stratified train/test split
        │
        ├──► K-Nearest Neighbors (KNN)
        ├──► Support Vector Machine (RBF)
        └──► Multinomial Naive Bayes
                    │
                    ▼
      Accuracy • Precision • Recall • F1
                    │
                    ▼
     Compare results and predict messages
```

### Models in the project

-   **K-Nearest Neighbors (KNN):** predicts a class using nearby
    examples in the processed feature space.
-   **Support Vector Machine (SVM, RBF kernel):** learns a decision
    boundary and supports probability estimates in this implementation.
-   **Multinomial Naive Bayes:** applies a probabilistic classification
    approach commonly used for text-related tasks.

### Text and numerical features

The pipeline uses TF-IDF text features with unigrams and bigrams,
alongside numeric fields such as:

-   Message length and word count
-   Number of URLs, digits, and special characters
-   Spam and legitimate keyword scores
-   Sender activity score and account age
-   Messages sent in the last 24 hours
-   Hour of day and day of week

## 📑 Dataset Requirements

The uploaded Excel/CSV file must contain the following columns. Column
names are case-sensitive after whitespace is stripped.

  Column                      Description
  --------------------------- ---------------------------------------------
  `message_text`              Message content
  `spam_label`                Target label: `0` = legitimate, `1` = spam
  `message_length`            Number of characters in the message
  `word_count`                Number of words
  `num_urls`                  Number of URLs
  `num_digits`                Number of digits
  `num_special_chars`         Number of special characters
  `spam_keyword_score`        Spam-related keyword score
  `legit_keyword_score`       Legitimate-message keyword score
  `sender_activity_score`     Sender activity estimate
  `sender_account_age_days`   Sender account age in days
  `messages_sent_last_24h`    Messages sent during the last 24 hours
  `hour_of_day`               Hour associated with the message
  `day_of_week`               Day of the week associated with the message

The loader also tolerates extra columns, but the listed fields are
required. The target must contain both classes (`0` and `1`).

## 📏 Evaluation Metrics

  -----------------------------------------------------------------------
  Metric                              What it tells you
  ----------------------------------- -----------------------------------
  **Accuracy**                        Fraction of all predictions that
                                      are correct

  **Precision**                       Of messages predicted as spam, how
                                      many were spam

  **Recall**                          Of actual spam messages, how many
                                      were identified

  **F1 score**                        Harmonic mean of precision and
                                      recall

  **Confusion matrix**                Counts of correct and incorrect
                                      predictions by class
  -----------------------------------------------------------------------

For spam detection, accuracy alone may be misleading---especially if the
dataset is imbalanced. Review precision and recall together, and check
false positives (legitimate messages flagged as spam) and false
negatives (spam messages missed).

## 🔬 Notes and Limitations

-   Model scores can vary with the dataset and selected test-set
    percentage.
-   The app currently uses a stratified hold-out split; it does not
    perform cross-validation or hyperparameter optimization.
-   For typed messages, keyword scores are estimated using a simple
    keyword list. For reliable production evaluation, use the same
    feature-generation logic used to create the training dataset.
-   Probability estimates are model estimates, not certainty.
-   Do not upload sensitive or private messages unless you have
    permission and understand how the data is handled.
-   Review false positives and false negatives before considering any
    real-world use.

## 🛠️ Troubleshooting

**The app says the dataset is missing**

-   Check the default dataset path described above, or upload the file
    in the sidebar.
-   Make sure the dataset contains every required column.

**Dataset validation fails**

-   Confirm that `spam_label` contains only `0` and `1`, and that both
    classes are present.
-   Check that numeric feature columns contain valid numeric values or
    missing values that can be imputed.

**A prediction cannot be made**

-   Train the models first in the **Model Comparison** tab.
-   Enter a non-empty message and verify that training completed
    successfully.

**A dependency is missing**

``` bash
pip install -r requirements.txt
```

## 🗺️ Future Improvements

-   Add cross-validation and hyperparameter tuning.
-   Add ROC-AUC and precision-recall curves.
-   Improve feature engineering and make training/inference
    preprocessing fully consistent.
-   Add automated tests and a reproducible evaluation report.
-   Improve model persistence and provide an option to load previously
    trained models.
-   Add model cards and clearer explanations for individual predictions.

## 🤝 Contributing

Ideas, bug reports, and improvements are welcome.

1.  Fork the repository.
2.  Create a branch: `git checkout -b feature/your-feature`.
3.  Commit your changes with a clear message.
4.  Push the branch and open a pull request.

## 📜 License

No license file was included in the supplied project files. Add a
`LICENSE` file before granting others permission to reuse, modify, or
distribute this project.

------------------------------------------------------------------------

## 👨‍💻 Author

**Meet Mehta**

-   GitHub: [@Meet290710](https://github.com/Meet290710)

### 🌟 If you find this project useful, consider giving the repository a star!

**Built for learning, experimentation, and better understanding of
machine-learning classification.**
:::
