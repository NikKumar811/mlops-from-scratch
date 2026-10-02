import argparse
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n-estimators",
    type=int,
    default=100
)

parser.add_argument(
    "--max-depth",
    type=int,
    default=None
)

args = parser.parse_args()


# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

MLFLOW_TRACKING_URI = "http://172.31.223.125:5000"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    "Iris Classification"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

print("Loading dataset...")

iris = load_iris()

X = iris.data
y = iris.target

print(f"Total records: {len(X)}")


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# MLflow Run
# --------------------------------------------------

with mlflow.start_run():

    print("Starting MLflow run...")


    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": 42,
        "test_size": 0.2
    })


    # ----------------------------------------------
    # Tags
    # ----------------------------------------------

    mlflow.set_tags({
        "project": "mlops-from-scratch",
        "dataset": "iris",
        "algorithm": "RandomForest"
    })


    # ----------------------------------------------
    # Train
    # ----------------------------------------------

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")


    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    print("Running inference on test data...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")


    # ----------------------------------------------
    # Metric
    # ----------------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    # ----------------------------------------------
    # Local Model
    # ----------------------------------------------

    print("Saving model...")

    joblib.dump(
        model,
        "model.pkl"
    )

    print("Model saved as model.pkl")


    # ----------------------------------------------
    # Training Summary Artifact
    # ----------------------------------------------

    with open(
        "training_summary.txt",
        "w"
    ) as file:

        file.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"n_estimators: {args.n_estimators}\n"
        )

        file.write(
            f"max_depth: {args.max_depth}\n"
        )


    mlflow.log_artifact(
        "training_summary.txt"
    )


    # ----------------------------------------------
    # Log Model
    # ----------------------------------------------

    print("Logging model to MLflow...")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="iris-model"
    )

    print(
        f"Model logged: {model_info.model_uri}"
    )


print("Experiment completed successfully!")
