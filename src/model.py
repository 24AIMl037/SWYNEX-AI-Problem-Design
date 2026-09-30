import os
import sys
from predict import EduIntentPredictor

def load_trained_model(models_dir: str = None):
    """
    Utility helper to instantiate trained EduIntentPredictor model.
    """
    return EduIntentPredictor(models_dir=models_dir)

if __name__ == "__main__":
    model = load_trained_model()
    sample = "Explain overfitting and bias variance tradeoff in Machine Learning"
    print("Test prediction:", model.predict(sample))
