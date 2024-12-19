from data_preprocessing import x_train, y_train, x_test, y_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix,
    classification_report
)

model = LogisticRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
