
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "genz_slang.csv"


def load_training_data():

    df = pd.read_csv(DATA_PATH)

    required = ["example_sentence", "category"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    df["example_sentence"] = (
        df["example_sentence"].astype(str).str.strip()
    )

    df["category"] = (
        df["category"].astype(str).str.strip()
    )

    df = df[
        (df["example_sentence"] != "") &
        (df["category"] != "")
    ]

    counts = df["category"].value_counts()

    # Keep categories with at least two examples
    df = df[
        df["category"].isin(counts[counts >= 2].index)
    ]

    return df


def train_model():

    df = load_training_data()

    if df["category"].nunique() < 2:
        raise ValueError(
            "At least two categories are required."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["example_sentence"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"]
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )
        ),
        (
            "svm",
            LinearSVC(
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Check whether cross-validation is possible
    train_counts = y_train.value_counts()
    min_class_count = train_counts.min()

    if min_class_count >= 2:

        cv_folds = min(5, int(min_class_count))

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )

        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [3000, 5000],
            "svm__C": [0.1, 1, 5, 10]
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            error_score="raise"
        )

        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_

        print("Best parameters:", grid_search.best_params_)
        print("Best CV Macro F1:", grid_search.best_score_)

    else:

        # Fallback when minority classes have too few examples
        print(
            "GridSearchCV skipped: insufficient training "
            "examples in minority categories."
        )

        model = pipeline
        model.fit(X_train, y_train)

    # Evaluate using the untouched test set
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    labels = sorted(df["category"].unique())

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame([
        {
            "Category": label,
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1-Score": report[label]["f1-score"],
            "Support": int(report[label]["support"])
        }
        for label in labels
    ])

    macro_f1 = report["macro avg"]["f1-score"]

    weighted_f1 = report["weighted avg"]["f1-score"]

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    return (
        model,
        accuracy,
        report_df,
        matrix_df,
        len(X_train),
        len(X_test),
        macro_f1,
        weighted_f1
    )


def predict_category(model, sentence):

    if not sentence or not sentence.strip():
        return None

    return model.predict([sentence.strip()])[0]
