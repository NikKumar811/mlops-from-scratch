import joblib

print("Loading trained model...")

model = joblib.load("model.pkl")

#sample = [[5.1, 3.5, 1.4, 0.2]]
sample = [[6.7, 3.0, 5.2, 2.3]]
prediction = model.predict(sample)

species = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

print("Prediction:", species[prediction[0]])
