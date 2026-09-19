import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Netflix Content Classifier",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CARGA DE DATOS
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
# CONFIGURACIÓN DEL MODELO
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


    X = model_df.drop(columns=[target])
    y = model_df[target]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # Modelo
    tree_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    tree_model.fit(X_train, y_train)


    return tree_model, encoders


tree_model, encoders = train_model(df)


# ============================================================
# ENCABEZADO
# ============================================================

st.title("🎬 Netflix Content Classifier")

st.markdown(
    """
    ### ¿Qué hace este modelo?

    Este proyecto utiliza **Machine Learning** para clasificar
    automáticamente contenido de Netflix.

    A partir de diferentes características del contenido,
    el modelo **Decision Tree** determina si se trata de:

    - 🎬 **Movie** — Película
    - 📺 **TV Show** — Serie de televisión

    La aplicación permite introducir las características de
    un contenido y obtener una predicción generada por el modelo.
    """
)


st.divider()


# ============================================================
# SECCIÓN DE PREDICCIÓN
# ============================================================

st.header("🔮 Realizar una predicción")

st.write(
    "Introduce las características del contenido que deseas clasificar."
)


# ============================================================
# DATOS DEL CONTENIDO
# ============================================================

st.subheader("📋 Características del contenido")


col1, col2 = st.columns(2)


with col1:

    release_year = st.number_input(
        "📅 Año de lanzamiento",
        min_value=1900,
        max_value=2030,
        value=2020,
        step=1
    )


    year_added = st.number_input(
        "📅 Año en que se agregó a Netflix",
        min_value=2000,
        max_value=2030,
        value=2020,
        step=1
    )


    month_added = st.slider(
        "📆 Mes en que se agregó",
        min_value=1,
        max_value=12,
        value=6
    )


    duration_num = st.number_input(
        "⏱️ Duración",
        min_value=1.0,
        max_value=500.0,
        value=90.0,
        step=1.0
    )


    director_count = st.number_input(
        "🎥 Número de directores",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )


    cast_count = st.number_input(
        "🎭 Número de actores",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


with col2:

    country_count = st.number_input(
        "🌎 Número de países involucrados",
        min_value=0,
        max_value=50,
        value=1,
        step=1
    )


    content_age = st.number_input(
        "📆 Antigüedad del contenido",
        min_value=0,
        max_value=150,
        value=5,
        step=1
    )


    is_multicountry = st.selectbox(
        "🌎 ¿Es una producción multinacional?",
        ["No", "Sí"]
    )


    is_long_content = st.selectbox(
        "⏱️ ¿Es contenido de larga duración?",
        ["No", "Sí"]
    )


    main_genre = st.selectbox(
        "🎭 Género principal",
        encoders["main_genre"].classes_
    )


    rating = st.selectbox(
        "🔞 Clasificación",
        encoders["rating"].classes_
    )


# ============================================================
# BOTÓN DE PREDICCIÓN
# ============================================================

st.divider()


col_button1, col_button2, col_button3 = st.columns(
    [1, 2, 1]
)


with col_button2:

    predict_button = st.button(
        "🔮 CLASIFICAR CONTENIDO",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICCIÓN
# ============================================================

if predict_button:

    # Convertir Sí / No a 1 / 0

    multicountry_value = (
        1 if is_multicountry == "Sí" else 0
    )

    long_content_value = (
        1 if is_long_content == "Sí" else 0
    )


    # Codificar variables categóricas

    genre_encoded = (
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

        "is_multicountry": multicountry_value,

        "is_long_content": long_content_value,

        "main_genre": genre_encoded,

        "rating": rating_encoded

    }])


    # ========================================================
    # PREDICCIÓN
    # ========================================================

    prediction = tree_model.predict(
        input_data
    )


    predicted_class = (
        encoders["type"]
        .inverse_transform(prediction)[0]
    )


    # ========================================================
    # PROBABILIDAD
    # ========================================================

    probabilities = (
        tree_model.predict_proba(input_data)[0]
    )

    classes = encoders["type"].classes_


    probability_dict = dict(
        zip(classes, probabilities)
    )


    confidence = max(probabilities)


    # ========================================================
    # RESULTADO
    # ========================================================

    st.divider()

    st.header("🎯 Resultado de la predicción")


    if predicted_class == "Movie":

        # Imagen de película
        image_url = (
            "https://images.unsplash.com/"
            "photo-1489599849927-2ee91cede3ba"
            "?auto=format&fit=crop&w=1200&q=80"
        )


        st.success(
            "🎬 El modelo predice que el contenido es una **PELÍCULA**"
        )


        col1, col2 = st.columns([1.5, 1])


        with col1:

            st.image(
                image_url,
                caption="🎬 Movie",
                use_container_width=True
            )


        with col2:

            st.metric(
                "Predicción",
                "🎬 MOVIE"
            )

            st.metric(
                "Confianza del modelo",
                f"{confidence:.1%}"
            )


    else:

        # Imagen de serie
        image_url = (
            "https://images.unsplash.com/"
            "photo-1522869635100-9f4c5e86aa37"
            "?auto=format&fit=crop&w=1200&q=80"
        )


        st.info(
            "📺 El modelo predice que el contenido es una "
            "**SERIE DE TELEVISIÓN**"
        )


        col1, col2 = st.columns([1.5, 1])


        with col1:

            st.image(
                image_url,
                caption="📺 TV Show",
                use_container_width=True
            )


        with col2:

            st.metric(
                "Predicción",
                "📺 TV SHOW"
            )

            st.metric(
                "Confianza del modelo",
                f"{confidence:.1%}"
            )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    st.subheader("📊 Probabilidad de clasificación")


    probability_df = pd.DataFrame({

        "Tipo de contenido": classes,

        "Probabilidad": probabilities

    })


    probability_df["Probabilidad"] = (
        probability_df["Probabilidad"] * 100
    )


    st.bar_chart(
        probability_df.set_index(
            "Tipo de contenido"
        )
    )


    # ========================================================
    # DATOS INGRESADOS
    # ========================================================

    with st.expander(
        "🔎 Ver características ingresadas"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Proyecto de Machine Learning — Netflix Content Classification"
)
