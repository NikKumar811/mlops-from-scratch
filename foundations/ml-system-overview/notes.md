# Video 1 — ML as a System

## 1. Traditional vs ML Systems

### Traditional
- Input → Code → Output
- Deterministic (same input → same output)
- Logic is explicitly written

### Machine Learning
- Data → Training → Model → Inference → Prediction
- Probabilistic output
- Model learns patterns from data

---

## 2. ML Pipeline

Dataset → Training → Model → Inference → Prediction

- Dataset: historical data
- Training: learning patterns
- Model: learned representation
- Inference: applying model
- Prediction: final output

---

## 3. Probabilistic Nature

- Output is not fixed
- Model gives probabilities (e.g., 87% fraud)
- Results are based on likelihood

---

## 4. Why Outputs Change

- Changing data changes model
- Retraining can change results
- Randomness affects output

---

## 5. Data Defines Behavior

- In traditional systems → code defines behavior
- In ML systems → data defines behavior

Key idea:
Changing data = changing behavior

---

## 6. Inference in Production

### Batch Inference
- Large dataset
- Runs on schedule
- Example: nightly fraud detection

### Real-time Inference
- API-based
- Instant response
- Example: payment fraud check
