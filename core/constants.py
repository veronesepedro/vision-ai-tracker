import os

# Caminhos base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Caminhos para os modelos
MP_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_recognizer.task")
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

