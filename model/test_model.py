import joblib
from pathlib import Path

model_path = Path("model")/"health_model.joblib"

if not model_path.exists():
    print("❌ Model file not found!")
    print("Expected:")
    print(model_path)
    exit()

bundle = joblib.load(model_path)

print("✅ Model loaded successfully!")

print("\nModel name:")
print(bundle["model_name"])

print("\nAccuracy:")
print(f"{bundle['accuracy']:.2%}")

print("\nNumber of features:")
print(len(bundle["features"]))

print("\nNumber of diseases:")
print(len(bundle["classes"]))