from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")

iris = load_iris()

X = iris.data
y = iris.target

print(f"Total records: {len(X)}")

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("Training model...")

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("Training completed!")

print("Running inference on test data...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

print("Saving model...")

joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
