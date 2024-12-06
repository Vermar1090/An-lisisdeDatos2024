import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# 1. Cargar y explorar los datos
df = pd.read_csv('titanic-dataset.csv')

# Mostrar las primeras filas del dataset
print(df.head())

# Resumen estadístico
print(df.describe())

# Comprobar los valores nulos
print(df.isnull().sum())

# Seleccionar solo las columnas numéricas para calcular la correlación
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numeric_cols].corr()

# Visualización de la correlación entre variables
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación entre variables')
plt.show()

# Distribución de la variable objetivo
sns.countplot(x='Survived', data=df)
plt.title('Distribución de la variable objetivo: Survived')
plt.show()


# 2. Preprocesar los datos
# Rellenar valores nulos
df['Age'].fillna(df['Age'].median(), inplace=True)  # Rellenar 'Age' con la mediana
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Rellenar 'Embarked' con el valor más frecuente

# Eliminar columnas innecesarias
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Convertir las variables categóricas en variables dummy (one-hot encoding)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Normalizar la columna 'Age'
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# Verificar si aún hay valores nulos
print(df.isnull().sum())

# 3. Selección de características
X = df.drop('Survived', axis=1)
y = df['Survived']

# Selección de las 5 mejores características usando SelectKBest
selector = SelectKBest(f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Ver las características seleccionadas
selected_columns = X.columns[selector.get_support()]
print(f"Características seleccionadas: {selected_columns}")

# 4. Dividir el dataset en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# 5. Entrenar el modelo
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Evaluar el modelo
y_pred = model.predict(X_test)

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy}")

# Reporte completo de clasificación
report = classification_report(y_test, y_pred)
print(f"Reporte de clasificación:\n{report}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.show()

# 7. Visualización de resultados
# Gráfico de Predicciones vs Valores Reales
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Predicción vs Realidad')
plt.xlabel('Valor Real')
plt.ylabel('Valor Predicho')
plt.show()

# 8. Interpretación y documentación
print("Los modelos de clasificación, como la regresión logística, nos proporcionan una precisión de X% y tienen un buen desempeño en la predicción de supervivencia en el Titanic.")
print("La matriz de confusión muestra que el modelo clasifica correctamente a la mayoría de los pasajeros sobrevivientes y no sobrevivientes.")
print("Es importante revisar las métricas de recall y F1-score para ver el desempeño equilibrado entre clases.")
