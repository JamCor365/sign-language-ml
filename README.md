# 🤟 Proyecto de Reconocimiento de Lengua de Señas en Español (LSM)

Este proyecto permite reconocer letras y palabras de la Lengua de Señas Mexicana (LSM) a partir de imágenes o video. Utiliza técnicas de Machine Learning y visión computacional, con modelos entrenados sobre letras estáticas y dinámicas.

## 📁 Estructura del proyecto

sign_language_project/
│
├── data/ # Datasets de letras y gestos
├── notebooks/ # Notebooks de análisis, entrenamiento y prueba
├── src/ # Scripts de procesamiento y modelos
├── models/ # Modelos entrenados (.pkl)
├── environment.yml # 🟢 Entorno Conda completo
├── requirements.txt # ⚪️ Requisitos para pip (opcional)
└── README.md

---

## ⚙️ Requisitos

- Python **3.11.x**
- [Git](https://git-scm.com/)
- Opcional: [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

---

## 🧪 Instalación del entorno

### 🔹 Opción 1: con **Conda** (recomendado)

1. **Clona este repositorio**:

```bash
git clone https://github.com/JamCor365/sign-language-ml.git
cd sign_language_project
```

2. **Crea el entorno Conda**:

```bash
conda env create -f environment.yml
conda activate signml
```

3. **(Opcional) Si usas Jupyter**:

```bash
python -m ipykernel install --user --name signml --display-name "Python (signml)"
```

### 🔹 Opción 2: con **pip** y entorno virtual de Python

1. **Crea entorno virtual manual**:

```bash
python3.11 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. **Instala dependencias desde `requirements.txt`**:

```bash
pip install -r requirements.txt
```
> Asegúrate de tener Python 3.11.x instalado previamente.

---

## 🚀 Uso

Abre los notebooks desde `notebooks/` para:

* `EDA_letters.ipynb`: Exploración y limpieza de letras
* `Training_letters.ipynb`: Entrenamiento de modelo estático
* `EDA_dynamics.ipynb`: Exploración y limpieza de secuencias dinámicas
* `Training_dynamics.ipynb`: Entrenamiento de modelo dinámico
* `predict_from_video.ipynb`: Prueba del sistema con un video de entrada
* `realtime_inference.ipynb`: Prueba en vivo con cámara web

---

## 🧠 Modelos utilizados

* HOG + PCA + SVM / Random Forest para letras
* Promedio de frames + PCA + Random Forest para gestos dinámicos
* Exploración con CNN + LSTM como modelo opcional (basado en [este artículo](https://www.nature.com/articles/s41598-024-76174-7))

