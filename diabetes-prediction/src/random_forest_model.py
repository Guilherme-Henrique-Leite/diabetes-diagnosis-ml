from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from data_preprocessing import x_train, y_train, x_test

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

y_pred_rf = rf_model.predict(x_test)
