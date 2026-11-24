# ⚖️ Plataforma Web de Detección y Mitigación de Sesgos de Género en Avisos de Empleo (Español)

Aplicación web interactiva desarrollada en **Python + Streamlit** para detectar, explicar y mitigar sesgos de género y discriminación en ofertas laborales en español mediante **Procesamiento de Lenguaje Natural (TF-IDF + Machine Learning)** y un motor de reescritura inclusiva que preserva íntegros los requisitos técnicos del puesto.

---

## 🎯 Funcionalidades Principales

1. **Dataset Sintético Balanceado en Español (`data/dataset_sesgos_empleo.csv`)**
   - **1,200 avisos de empleo únicos** perfectamente balanceados (**200 ejemplos por clase**).
   - **6 clases etiquetadas**:
     - `Masculino`: lenguaje androcéntrico o estereotipos hipercompetitivos/dominantes.
     - `Femenino`: estereotipos de feminidad tradicional, apariencia (`excelente presencia`, `buena presencia`), roles maternales/sumisos (`trato dulce`, `empático`, `maternal`, `sumisa`, `vocación de servicio`) o títulos feminizados (`asistente administrativa`, `recepcionista`, `chica ordenada`).
     - `Excluyente`: exclusión explícita o barreras por sexo, embarazo, estado de gestación, hijos, títulos con género exclusivo o responsabilidades familiares.
     - `LGBTQ+`: requisitos cisheteronormativos o restricciones sobre identidad/expresión de género, identidad no binaria o valores familiares convencionales.
     - `Demográfico`: filtros por edad, estado civil, nacionalidad, zona de residencia o fotografía corporal.
     - `Neutro`: redacción inclusiva, objetiva y centrada en competencias (incluye **60 ejemplos dedicados con cláusulas reales de inclusión** como *"sin distinción de género, edad ni orientación"*, *"igualdad de oportunidades"* y *"proceso inclusivo y diverso"* para evitar falsos positivos con `Excluyente`).
   - Distribución uniforme de **20 sectores laborales** y requisitos técnicos entre todas las clases para prevenir sesgo temático (*topic leakage*).

2. **Entrenamiento y Comparación de 3 Clasificadores ML con TF-IDF (`src/train_models.py`)**
   - Vectorización **TF-IDF** (unigramas y bigramas `(1, 2)`, `sublinear_tf=True`, normalización en español preservando negaciones, marcas de género y exclusividad).
   - Comparación de **3 modelos**:
     - **Random Forest** (`RandomForestClassifier`)
     - **SVM** (`CalibratedClassifierCV(SVC)`)
     - **Decision Tree** (`DecisionTreeClassifier`)
   - Exportación automática de métricas exactas (**Accuracy, Precision, Recall, F1-Score, Validación Cruzada 5-Fold y Matriz de Confusión**) en formato **JSON y CSV** dentro de `reports/`.

3. **Interfaz Interactiva en Streamlit (`app.py`)**
   - **Analizador y Mitigador**:
     - Cuadro de texto libre para pegar cualquier oferta laboral y **7 ejemplos precargados** (uno por cada categoría de sesgo, un caso mixto multi-sesgo y una oferta neutra).
     - **Resaltado visual** en línea de las frases sesgadas detectadas, con código de color por categoría, *tooltips* explicativos y tabla detallada de hallazgos.
     - **Gráfico interactivo de probabilidades** para las 5 categorías de sesgo más probabilidad de clase `Neutro` e Índice de Inclusividad.
     - **Botón de reescritura neutra e inclusiva**: genera automáticamente una versión mitigada, verifica al 100% la preservación de los requisitos técnicos detectados (lenguajes, herramientas, certificaciones, idiomas, años de experiencia) y re-evalúa en vivo la mejora con el modelo ML.
   - **Comparativa de Modelos ML**: tabla comparativa, gráficos de barras, matrices de confusión interactivas, reporte por clase y descarga directa de archivos JSON/CSV para informes.
   - **Explorador del Dataset**: gráficos de balance de clases, filtros por categoría/sector/palabra clave y descarga del CSV.

---

## 📁 Estructura del Proyecto

```text
sesgo de genero/
├── app.py                              # Aplicación web interactiva en Streamlit
├── requirements.txt                    # Dependencias de Python
├── README.md                           # Instrucciones de instalación, uso y métricas
├── src/
│   ├── __init__.py
│   ├── config.py                       # Categorías, paleta visual, rutas y ejemplos precargados
│   ├── text_utils.py                   # Normalización en español 1-a-1 y stopwords curadas
│   ├── generate_dataset.py             # Generador reproducible del dataset balanceado (1,200 ejemplos)
│   ├── train_models.py                 # Entrenamiento TF-IDF + Random Forest, SVM y Decision Tree
│   ├── bias_analyzer.py                # Inferencia ML, detección de frases sesgadas y resaltado HTML
│   └── rewriter.py                     # Motor de reescritura inclusiva con validación técnica
├── tests/
│   └── test_pipeline.py                # Pruebas automatizadas end-to-end
├── data/
│   └── dataset_sesgos_empleo.csv       # Dataset sintético balanceado (1,200 avisos)
├── models/
│   ├── random_forest.joblib            # Pipeline TF-IDF + Random Forest
│   ├── svm.joblib                      # Pipeline TF-IDF + SVM Calibrado
│   ├── decision_tree.joblib            # Pipeline TF-IDF + Decision Tree
│   └── best_model_meta.json            # Metadatos del mejor modelo
└── reports/
    ├── model_metrics.json              # Métricas completas exactas + matrices de confusión (JSON)
    ├── model_metrics_summary.csv       # Tabla resumen comparativa de modelos (CSV)
    ├── per_class_metrics.csv           # Precision, Recall, F1 y soporte por clase (CSV)
    └── confusion_matrices.csv          # Matrices de confusión de los 3 modelos (CSV)
```

---

## ⚙️ Instalación

### 1. Requisitos previos
- **Python 3.10 o superior** instalado y disponible en el sistema.

### 2. Clonar o ubicarse en la carpeta del proyecto
```bash
cd "c:\Users\verde\OneDrive\Desktop\sesgo de genero"
```

### 3. (Opcional recomendado) Crear y activar un entorno virtual
En **Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
En **macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar las dependencias
```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución

> El repositorio ya incluye el dataset generado, los 3 modelos entrenados y los archivos de métricas en `reports/`. No obstante, puedes reproducir el pipeline completo en cualquier momento con los siguientes pasos:

### Paso 1 (opcional): Regenerar el dataset sintético balanceado
```bash
python -m src.generate_dataset
```

### Paso 2 (opcional): Entrenar modelos y regenerar reportes JSON/CSV
```bash
python -m src.train_models
```

### Paso 3 (opcional): Ejecutar las pruebas de validación
```bash
python -m tests.test_pipeline
```

### Paso 4: Iniciar la plataforma web en Streamlit
```bash
streamlit run app.py
```
Si `streamlit` no está en tu `PATH` en Windows, ejecútalo como módulo de Python:
```powershell
python -m streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## 📊 Resultados de Evaluación de Modelos (Para Informe)

Evaluación realizada sobre un split estratificado **80% entrenamiento (960 avisos) / 20% prueba (240 avisos, 40 por clase)** con semilla fija (`random_state=42`) y **validación cruzada estratificada de 5 pliegues (CV-5)** sobre entrenamiento:

| Modelo | Accuracy (Test) | Precision Macro | Recall Macro | F1-Score Macro | CV-5 Accuracy | CV-5 F1-Macro (± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** ⭐ | **0.9458** | **0.9471** | **0.9458** | **0.9458** | **0.9448** | **0.9454 (± 0.0144)** |
| **SVM** | 0.9375 | 0.9387 | 0.9371 | 0.9374 | 0.9000 | 0.9012 (± 0.0220) |
| **Decision Tree** | 0.9250 | 0.9302 | 0.9252 | 0.9260 | 0.8875 | 0.8874 (± 0.0317) |

- **Archivos listos para citar o importar en tablas/gráficos**:
  - `reports/model_metrics.json` — Estructura completa con métricas globales, por clase, CV-5 y matrices de confusión 6×6.
  - `reports/model_metrics_summary.csv` — Tabla comparativa directa de los 3 clasificadores.
  - `reports/per_class_metrics.csv` — Desempeño detallado por categoría (`Masculino`, `Femenino`, `Excluyente`, `LGBTQ+`, `Demográfico`, `Neutro`).
  - `reports/confusion_matrices.csv` — Matrices de confusión de los 3 modelos en formato tabular.

---

## 🛠️ Consideraciones Metodológicas

- **Stopwords curadas para sesgo**: A diferencia de listas genéricas de stopwords en español, el preprocesamiento (`src/text_utils.py`) conserva palabras clave portadoras de sesgo como negaciones (`no`, `sin`), marcadores de exclusividad (`solo`, `únicamente`, `exclusivamente`) y artículos/pronombres con marca de género.
- **Preservación de requisitos técnicos**: El motor de reescritura (`src/rewriter.py`) actúa exclusivamente sobre expresiones sesgadas o discriminatorias y verifica automáticamente que todos los términos técnicos extraídos (p. ej., `Python`, `SQL`, `AWS`, `SAP`, `Power BI`, `ISO 9001`, niveles de idioma y años de experiencia) permanezcan intactos en el texto final.
