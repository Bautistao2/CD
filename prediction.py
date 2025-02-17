import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cargar el modelo guardado
best_clf = joblib.load("models/arbol_decision.pkl")

# Elegir una muestra aleatoria
random_index = np.random.randint(0, X.shape[0])
sample = X_scaled[random_index].reshape(1, -1)

# Obtener la predicción del modelo
prediction = best_clf.predict(sample)[0]
prediction_label = "Maligno" if prediction == 0 else "Benigno"

# Obtener la ruta de decisión de la muestra
node_indicator = best_clf.decision_path(sample)
leaf_id = best_clf.apply(sample)[0]  # Nodo hoja final

# Aumentar el tamaño de la imagen para evitar sobreposición
plt.figure(figsize=(24, 12))

# Dibujar el árbol de decisión
tree_plot = plot_tree(best_clf, feature_names=data.feature_names, class_names=data.target_names, filled=True, rounded=True, fontsize=12, impurity=False, proportion=True)

# Obtener la lista de nodos
ax = plt.gca()

# Resaltar los nodos del camino de decisión con menos opacidad
for node_id in node_indicator.indices:
    if node_id < len(ax.texts):  # Verificamos que el nodo existe para evitar errores
        ax.texts[node_id].set_bbox(dict(facecolor="red", alpha=0.2, edgecolor="black", boxstyle="round,pad=0.2"))

# Guardar la imagen con la ruta resaltada
image_path = "models/arbol_decision_marcado.png"
plt.savefig(image_path, dpi=300, bbox_inches="tight")  # Guardar con mejor resolución y sin bordes innecesarios

# Imprimir resultados en la consola
print(f"🔍 Predicción para la muestra {random_index}: {prediction_label}")
print(f"📸 Imagen del árbol guardada en {image_path} (decisión en nodo {leaf_id})")
print(f"📌 La muestra pasó por los nodos: {list(node_indicator.indices)}")
print(f"✅ Nodo final de decisión: {leaf_id} ({prediction_label})")

# Mostrar la imagen mejorada
plt.show()
