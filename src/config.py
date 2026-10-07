"""Central configuration: paths, column groups and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "credit_card_fraud_10k.xlsx"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
TEST_PREDS_PATH = MODELS_DIR / "test_predictions.csv"

ID_COL = "transaction_id"
TARGET = "is_fraud"

NUMERIC = ["amount", "device_trust_score", "velocity_last_24h", "cardholder_age"]
CYCLIC = ["transaction_hour"]
BINARY = ["foreign_transaction", "location_mismatch"]
CATEGORICAL = ["merchant_category"]
FEATURES = NUMERIC + CYCLIC + BINARY + CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
C_GRID = [0.01, 0.1, 1.0, 10.0]
# Threshold tuning: maximise F-beta (beta=2 favours recall) subject to a precision floor.
F_BETA = 2.0
MIN_PRECISION = 0.30
