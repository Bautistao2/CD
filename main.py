import os

# Ejecutar el entrenamiento del modelo
print("Entrenando el modelo...")
os.system("python train_model.py")

# Ejecutar la validación del modelo
print("Evaluando el modelo...")
os.system("python test_model.py")

print("Proceso completado. Modelo entrenado y evaluado correctamente.")
