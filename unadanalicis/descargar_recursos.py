import os
import pandas as pd
import kagglehub

# Obtener el directorio de trabajo actual
working_directory = os.getcwd()
print(f"Directorio de trabajo actual: {working_directory}")

# Descargar el dataset
dataset_name = "yasserh/titanic-dataset"
path = kagglehub.dataset_download(dataset_name)

# Verificar si el dataset se descargó correctamente
dataset_folder = os.path.join(path, "train.csv")

# Si el archivo no existe en la ruta esperada, imprimimos un mensaje y terminamos
if not os.path.exists(dataset_folder):
    print(f"No se encontró el archivo 'train.csv' en la ruta esperada: {dataset_folder}")
else:
    # Mover el archivo al directorio de trabajo
    new_file_path = os.path.join(working_directory, "train.csv")
    os.rename(dataset_folder, new_file_path)
    print(f"El archivo 'train.csv' ha sido movido a: {new_file_path}")

    # Cargar el archivo CSV desde la nueva ubicación
    try:
        df = pd.read_csv(new_file_path)
        print("El archivo se cargó correctamente.")
        print(df.head())  # Mostrar las primeras filas del dataset
    except FileNotFoundError:
        print(f"No se pudo cargar el archivo 'train.csv' desde la ruta: {new_file_path}")
