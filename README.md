# CD
# Proyecto: Clasificación de Cáncer de Mama con Árbol de Decisión

Este proyecto implementa un **modelo de Árbol de Decisión** para la **clasificación de tumores de mama** utilizando el conjunto de datos de **Cáncer de Mama de Wisconsin**. Se entrena el modelo, se ajustan hiperparámetros con `GridSearchCV`, se evalúa con datos de prueba y se visualiza la estructura del árbol de decisión.

## 📂 **Estructura del Proyecto**

```
📂 models            # Carpeta donde se guardan los modelos entrenados
📂 notebooks         # Carpeta opcional para análisis exploratorio
📂 data              # Carpeta opcional para almacenar datasets
📜 main.py           # Script principal que ejecuta entrenamiento y evaluación
📜 train_model.py    # Script para entrenamiento del modelo y ajuste de hiperparámetros
📜 test_model.py     # Script para evaluar el modelo en los datos completos
📜 predictor.py      # Script para hacer predicciones individuales y visualizar el árbol de decisión
📜 README.md         # Documentación del proyecto
```

---

## 📊 **Dataset**
El dataset utilizado es el **Cáncer de Mama de Wisconsin**, disponible en `sklearn.datasets.load_breast_cancer()`. Contiene 569 muestras con **30 características** que describen propiedades del tumor, y una etiqueta de clasificación:

- **0 → Maligno**
- **1 → Benigno**

El dataset ya está preprocesado y listo para ser utilizado en modelos de Machine Learning.

---

## 🔧 **Entrenamiento del Modelo (`train_model.py`)**

### 🔹 **Preprocesamiento de Datos**
1. Se carga el dataset con `load_breast_cancer()`.
2. Se divide en **80% datos de entrenamiento** y **20% datos de prueba** con `train_test_split()`.
3. Se **escalan las características** con `StandardScaler()` para mejorar el rendimiento del modelo.

### 🔹 **Definición del Modelo y Búsqueda de Hiperparámetros**
Se entrena un **Árbol de Decisión (`DecisionTreeClassifier`)**, ajustando los hiperparámetros con `GridSearchCV`.

#### **📌 Hiperparámetros Evaluados:**
```python
param_grid = {
    'criterion': ['gini', 'entropy'],      # Función de evaluación del árbol
    'max_depth': [None, 10, 20, 30],       # Profundidad máxima del árbol
    'min_samples_split': [2, 5, 10],       # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4]          # Mínimo de muestras en cada hoja
}
```

### 🔹 **¿Cómo Funciona `GridSearchCV`?**
`GridSearchCV` busca la mejor combinación de hiperparámetros evaluando cada configuración con **validación cruzada (`cv=5`)**. Se utiliza `n_jobs=-1` para aprovechar todos los núcleos disponibles y **optimizar la búsqueda**.

```python
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)
```

📌 **Resultado:** Se selecciona el mejor modelo (`best_clf`), que luego se **guarda en `models/arbol_decision.pkl`** para futuras predicciones.

### 🔹 **Evaluación del Modelo**
Después del entrenamiento, se evalúa el modelo con los datos de prueba:
```python
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
```
Esto muestra la **precisión, recall, F1-score** y otras métricas clave.

---

## 🧪 **Evaluación del Modelo (`test_model.py`)**
Este script carga el modelo entrenado (`arbol_decision.pkl`) y lo evalúa con **todo el dataset**.

- Se cargan los datos y se **escalan nuevamente** con el mismo `StandardScaler()`.
- Se cargan los pesos del modelo con `joblib.load()`.
- Se hacen predicciones sobre `X_scaled`.
- Se generan métricas de evaluación con `classification_report()` y `confusion_matrix()`.

📌 **Esto verifica si el modelo generaliza bien a datos nuevos.**

---

## 🔍 **Predicciones y Visualización (`predictor.py`)**
Este script selecciona una muestra aleatoria, **predice su clase y muestra el árbol de decisión con los nodos resaltados**.

1. **Carga el modelo entrenado** (`arbol_decision.pkl`).
2. **Toma una muestra aleatoria** y obtiene su ruta en el árbol.
3. **Dibuja el árbol de decisión** con `plot_tree()`.
4. **Resalta los nodos del camino de decisión** en rojo.

```python
for node_id in node_indicator.indices:
    plt.gca().texts[node_id].set_bbox(dict(facecolor="red", alpha=0.2, edgecolor="black", boxstyle="round,pad=0.2"))
```

📸 **La imagen del árbol se guarda en `models/arbol_decision_marcado.png`.**

📌 **Esto permite visualizar cómo el modelo toma decisiones.**

---

## 🚀 **Ejecución del Proyecto**
### **1️⃣ Entrenar el Modelo**
```bash
python main.py
```
Esto entrenará el modelo, guardará los pesos y evaluará el rendimiento.

### **2️⃣ Hacer Predicciones**
```bash
python predictor.py
```
Esto imprimirá la **predicción de una muestra aleatoria** y generará la imagen del árbol con los nodos resaltados.

---

## 🛠 **Requisitos y Librerías**
Antes de ejecutar el proyecto, instala las dependencias con:
```bash
pip install -r requirements.txt
```

Librerías utilizadas:
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`

---

## 📌 **Conclusión**
Este proyecto muestra cómo entrenar, optimizar y evaluar un **Árbol de Decisión** para la clasificación de tumores de mama. Además, permite visualizar el proceso de decisión del modelo, lo que es útil para entender cómo toma sus predicciones.


