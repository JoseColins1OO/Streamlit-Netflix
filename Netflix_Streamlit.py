```python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Clasificador de contenido de Netflix",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# 2. TÍTULO
# ============================================================

st.title("🎬 Clasificador de contenido de Netflix")
st.markdown(
    """
    Esta aplicación utiliza un **Decision Tree** para clasificar
    contenido de Netflix como **Movie** o **TV Show**.
    """
)


# ============================================================
# 3. CARGA DE DATOS
# ============================================================

@st.cache_data
def load_data():

    linkdata = (
        "https://raw.githubusercontent.com/"
        "JoseColins1OO/Proyecto_Netflix/"
        "refs/heads/main/data/processed/netflix_clean.csv"
    )

    return pd.read_csv(linkdata)


df = load_data()


# ============================================================
# 4. FEATURES
# ============================================================

features = [
    "release_year",
    "year_added",
    "month_added",
    "duration_num",
    "director_count",
    "cast_count",
    "country_count",
    "content_age",
    "is_multicountry",
    "is_long_content",
    "main_genre",
    "rating"
]

target = "type"


# ============================================================
# 5. PREPARACIÓN DE DATOS
# ============================================================

@st.cache_resource
def train_model(df):

    model_df = df[features + [target]].copy()

    categorical_cols = [
        "main_genre",
        "rating",
        "type"
    ]

    encoders = {}

    # Codificación de variables categóricas
    for col in categorical_cols:

        le = LabelEncoder()

        model_df[col] = le.fit_transform(
            model_df[col].astype(str)
        )

        encoders[col] = le


    # Valores nulos
    model_df["year_added"] = (
        model_df["year_added"]
        .fillna(model_df["year_added"].median())
    )

    model_df["month_added"] = (
        model_df["month_added"]
        .fillna(model_df["month_added"].median())
    )


    # X / y
    X = model_df.drop(columns=[target])
    y = model_df[target]


    # Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # ========================================================
    # Decision Tree
    # ========================================================

    tree_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    tree_pred = tree_model.predict(X_test)


    return (
        model_df,
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        tree_model,
        tree_pred,
        encoders
    )


(
    model_df,
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
    tree_model,
    tree_pred,
    encoders
) = train_model(df)


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("📌 Menú")

option = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Inicio",
        "Datos",
        "Métricas del modelo",
        "Matriz de confusión",
        "Árbol de decisión",
        "Realizar predicción"
    ]
)


# ============================================================
# 7. INICIO
# ============================================================

if option == "Inicio":

    st.header("🎬 Clasificador de contenido Netflix")

    st.write(
        """
        El objetivo de este proyecto es desarrollar un modelo de
        Machine Learning capaz de determinar si un contenido de
        Netflix corresponde a una **película (Movie)** o una
        **serie de televisión (TV Show)**.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Registros",
            len(df)
        )

    with col2:
        st.metric(
            "Características",
            len(features)
        )

    with col3:
        st.metric(
            "Profundidad del árbol",
            tree_model.max_depth
        )


    st.subheader("Distribución del contenido")

    content_counts = df[target].value_counts()

    st.bar_chart(content_counts)


# ============================================================
# 8. DATOS
# ============================================================

elif option == "Datos":

    st.header("📊 Dataset de Netflix")

    st.write(
        f"El dataset contiene **{len(df):,} registros**."
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Características utilizadas")

    st.write(features)

    st.subheader("Distribución del target")

    st.dataframe(
        df[target].value_counts()
    )


# ============================================================
# 9. MÉTRICAS
# ============================================================

elif option == "Métricas del modelo":

    st.header("📈 Evaluación del modelo")

    accuracy = accuracy_score(
        y_test,
        tree_pred
    )

    precision = precision_score(
        y_test,
        tree_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        tree_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        tree_pred,
        zero_division=0
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1:.2%}"
        )


    st.subheader("Benchmark")

    benchmark = pd.DataFrame({
        "Model": ["Decision Tree"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1]
    })

    st.dataframe(
        benchmark,
        use_container_width=True
    )


# ============================================================
# 10. MATRIZ DE CONFUSIÓN
# ============================================================

elif option == "Matriz de confusión":

    st.header("🔢 Matriz de Confusión")

    cm = confusion_matrix(
        y_test,
        tree_pred
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoders["type"].classes_
    ).plot(
        ax=ax
    )

    ax.set_title(
        "Decision Tree - Confusion Matrix"
    )

    st.pyplot(fig)


# ============================================================
# 11. ÁRBOL DE DECISIÓN
# ============================================================

elif option == "Árbol de decisión":

    st.header("🌳 Árbol de Decisión")

    st.write(
        """
        El siguiente árbol muestra las reglas utilizadas por
        el modelo para clasificar el contenido.
        """
    )

    fig, ax = plt.subplots(
        figsize=(24, 12)
    )

    plot_tree(
        tree_model,
        feature_names=X.columns,
        class_names=encoders["type"].classes_,
        filled=True,
        rounded=True,
        ax=ax
    )

    ax.set_title(
        "Decision Tree Classification Rules"
    )

    st.pyplot(fig)


# ============================================================
# 12. PREDICCIÓN
# ============================================================

elif option == "Realizar predicción":

    st.header("🔮 Realizar una predicción")

    st.write(
        """
        Introduce las características del contenido para que
        el modelo determine si corresponde a una película
        o una serie de televisión.
        """
    )


    # --------------------------------------------------------
    # Variables numéricas
    # --------------------------------------------------------

    st.subheader("📊 Características numéricas")

    col1, col2 = st.columns(2)

    with col1:

        release_year = st.number_input(
            "Año de lanzamiento",
            min_value=1900,
            max_value=2030,
            value=2020,
            step=1
        )

        year_added = st.number_input(
            "Año agregado a Netflix",
            min_value=2000,
            max_value=2030,
            value=2020,
            step=1
        )

        month_added = st.number_input(
            "Mes agregado",
            min_value=1,
            max_value=12,
            value=1,
            step=1
        )

        duration_num = st.number_input(
            "Duración",
            min_value=1.0,
            max_value=500.0,
            value=90.0,
            step=1.0
        )

        director_count = st.number_input(
            "Número de directores",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        cast_count = st.number_input(
            "Número de actores",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )


    with col2:

        country_count = st.number_input(
            "Número de países",
            min_value=0,
            max_value=50,
            value=1,
            step=1
        )

        content_age = st.number_input(
            "Antigüedad del contenido",
            min_value=0,
            max_value=150,
            value=5,
            step=1
        )

        is_multicountry = st.selectbox(
            "¿Es producción multinacional?",
            [0, 1]
        )

        is_long_content = st.selectbox(
            "¿Es contenido largo?",
            [0, 1]
        )


    # --------------------------------------------------------
    # Variables categóricas
    # --------------------------------------------------------

    st.subheader("🏷️ Características categóricas")


    main_genre_options = (
        list(encoders["main_genre"].classes_)
    )

    rating_options = (
        list(encoders["rating"].classes_)
    )


    col1, col2 = st.columns(2)

    with col1:

        main_genre = st.selectbox(
            "Género principal",
            main_genre_options
        )


    with col2:

        rating = st.selectbox(
            "Clasificación",
            rating_options
        )


    # --------------------------------------------------------
    # Botón
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🔮 Clasificar contenido",
        type="primary"
    ):

        # Codificar variables categóricas

        main_genre_encoded = (
            encoders["main_genre"]
            .transform([main_genre])[0]
        )

        rating_encoded = (
            encoders["rating"]
            .transform([rating])[0]
        )


        # Crear DataFrame

        input_data = pd.DataFrame([{

            "release_year": release_year,

            "year_added": year_added,

            "month_added": month_added,

            "duration_num": duration_num,

            "director_count": director_count,

            "cast_count": cast_count,

            "country_count": country_count,

            "content_age": content_age,

            "is_multicountry": is_multicountry,

            "is_long_content": is_long_content,

            "main_genre": main_genre_encoded,

            "rating": rating_encoded
        }])


        # Predicción

        prediction = tree_model.predict(
            input_data
        )


        # Convertir predicción a texto

        predicted_class = (
            encoders["type"]
            .inverse_transform(prediction)[0]
        )


        # Probabilidades

        probabilities = (
            tree_model
            .predict_proba(input_data)[0]
        )


        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        st.subheader("🎯 Resultado")

        if predicted_class == "Movie":

            st.success(
                "🎬 El modelo predice: **MOVIE**"
            )

        else:

            st.info(
                "📺 El modelo predice: **TV SHOW**"
            )


        # ----------------------------------------------------
        # Probabilidades
        # ----------------------------------------------------

        st.subheader("📊 Probabilidad de clasificación")


        probability_df = pd.DataFrame({

            "Clase":
                encoders["type"].classes_,

            "Probabilidad":
                probabilities

        })


        st.dataframe(
            probability_df,
            use_container_width=True
        )


        st.bar_chart(
            probability_df.set_index("Clase")
        )


        # ----------------------------------------------------
        # Datos introducidos
        # ----------------------------------------------------

        with st.expander(
            "Ver datos utilizados para la predicción"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "Proyecto de Machine Learning - Netflix"
)
```

