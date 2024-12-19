import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = ['NumGravidezes', 'Glicose', 'PressaoSanguinea', 'EspessuraPele',
           'Insulina', 'IMC', 'HistoricoDiabetes', 'Idade', 'Resultado']

df = pd.read_csv(URL, header=None, names=columns)

df.head()
df.fillna(df.median(), inplace=True)

x = df.drop('Resultado', axis=1)
y = df['Resultado']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)

x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print(f"Size of X_train_smote: {x_train_smote.shape}")
print(f"Size of y_train_smote: {y_train_smote.shape}")
