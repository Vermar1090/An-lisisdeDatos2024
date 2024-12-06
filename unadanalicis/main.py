import tkinter as tk
from tkinter import messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

# Función para cargar y procesar el dataset
def cargar_datos():
    global df
    try:
        # Cargar el dataset
        df = pd.read_csv('titanic-dataset.csv')
        
        # Llenar los valores nulos de 'Age' y 'Embarked'
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        # Eliminar columnas innecesarias
        df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        
        # Codificar variables categóricas
        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
        
        # Normalizar la columna 'Age'
        scaler = StandardScaler()
        df['Age'] = scaler.fit_transform(df[['Age']])
        
        print("Datos cargados y preprocesados exitosamente.")
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un error cargando los datos: {str(e)}")

# Función para evaluar y entrenar el modelo
def evaluar_modelo(model, modelo_nombre):
    try:
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if modelo_nombre == 'Regresión Logística' or modelo_nombre == 'Árbol de Decisión':
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matriz de Confusión ({modelo_nombre})')
            plt.show()
            messagebox.showinfo("Resultados", f"Precisión: {accuracy}\nReporte de Clasificación:\n{report}")
        
        elif modelo_nombre == 'Regresión Lineal':
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            sns.scatterplot(x=y_test, y=y_pred)
            plt.title('Regresión Lineal: Predicción vs Realidad')
            plt.xlabel('Valor Real')
            plt.ylabel('Valor Predicho')
            plt.show()
            messagebox.showinfo("Resultados", f"R^2: {r2}\nError cuadrático medio: {mse}")
    except Exception as e:
        messagebox.showerror("Error", f"Hubo un error durante la evaluación del modelo: {str(e)}")

# Función para seleccionar el modelo y ejecutar el flujo
def seleccionar_modelo():
    modelo_seleccionado = modelo_var.get()
    if not modelo_seleccionado:
        messagebox.showwarning("Advertencia", "Por favor selecciona un modelo.")
        return

    cargar_datos()  # Cargar los datos antes de entrenar el modelo
    if modelo_seleccionado == "Logística":
        model = LogisticRegression(max_iter=200)
        evaluar_modelo(model, "Regresión Logística")
    elif modelo_seleccionado == "Lineal":
        model = LinearRegression()
        evaluar_modelo(model, "Regresión Lineal")
    elif modelo_seleccionado == "Árbol de Decisión":
        model = DecisionTreeClassifier(random_state=42)
        evaluar_modelo(model, "Árbol de Decisión")

# Crear la ventana principal
root = tk.Tk()
root.title("Juan Vergara Martinez Tarea 5 Proyecto Análisis de Datos")

# Configurar tamaño de la ventana
root.geometry("400x300")

# Título
titulo = tk.Label(root, text="Selecciona un Modelo de Aprendizaje Supervisado", font=("Arial", 14))
titulo.pack(pady=20)

# Crear variable para la selección del modelo
modelo_var = tk.StringVar()

# Botones para seleccionar el modelo
modelo_logistica = tk.Radiobutton(root, text="Regresión Logística", variable=modelo_var, value="Logística")
modelo_logistica.pack(anchor="w", padx=20)
modelo_lineal = tk.Radiobutton(root, text="Regresión Lineal", variable=modelo_var, value="Lineal")
modelo_lineal.pack(anchor="w", padx=20)
modelo_arbol = tk.Radiobutton(root, text="Árbol de Decisión", variable=modelo_var, value="Árbol de Decisión")
modelo_arbol.pack(anchor="w", padx=20)

# Botón para ejecutar el análisis
boton_ejecutar = tk.Button(root, text="Ejecutar Análisis", command=seleccionar_modelo)
boton_ejecutar.pack(pady=20)

# Ejecutar la interfaz gráfica
root.mainloop()
