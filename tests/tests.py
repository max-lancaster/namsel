import joblib
import os

def check_scaler():
    SCALER_PATH = 'zernike_scaler-latest'
    assert os.path.exists(SCALER_PATH)

    scaler = joblib.load(SCALER_PATH)
    
    assert hasattr(scaler, 'scale_') # Make sure not using a deprecated version of sklearn
