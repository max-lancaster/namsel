import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Dummy data for fitting
X = np.random.random((100, 512))

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, "zernike_scaler-latest")
