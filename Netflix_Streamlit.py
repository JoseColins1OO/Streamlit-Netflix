
# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Netflix Content Classifier",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CARGAR DATASET
# ============================================================

@st.cache_data
def load_data():

    linkdata = (
        "https://raw.githubusercontent.com/"
        "JoseColins1OO/Proyecto_Netflix/"
        "refs/heads/main/data/processed/netflix_clean.csv"
    )

    df = pd.read_csv(linkdata)

    return df


df = load_data()


# ============================================================
# PREPROCESAMIENTO DE DURACIÓN
# ============================================================

def prepare_duration(df):

    df = df.copy()

    # Normalizar texto de duration_unit
    df["duration_unit"] = (
        df["duration_unit"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convertir temporadas a una escala aproximada
    #
    # Movie:
    #   90 min  -> 90
    #   120 min -> 120
    #
    # TV Show:
    #   1 season -> 30
    #   2 seasons -> 60
    #   3 seasons -> 90
    #
    # Esto permite que el modelo trabaje con una escala
    # comparable entre películas y series.

    df["duration_model"] = df.apply(
        lambda row:
            row["duration_num"]
            if row["duration_unit"] in ["min", "mins", "minute", "minutes"]
            else row["duration_num"] * 30,
        axis=1
    )

    return df


df = prepare_duration(df)


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

features = [
    "release_year",
    "year_added",
    "month_added",
    "duration_model",
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
# ENTRENAMIENTO
# ============================================================

@st.cache_resource
def train_model(df):

    model_df = df[features + [target]].copy()

    # --------------------------------------------------------
    # CODIFICAR VARIABLES CATEGÓRICAS
    # --------------------------------------------------------

    categorical_cols = [
        "main_genre",
        "rating",
        "type"
    ]

    encoders = {}

    for col in categorical_cols:

        encoder = LabelEncoder()

        model_df[col] = encoder.fit_transform(
            model_df[col].astype(str)
        )

        encoders[col] = encoder


    # --------------------------------------------------------
    # VALORES FALTANTES
    # --------------------------------------------------------

    numeric_columns = [
        "release_year",
        "year_added",
        "month_added",
        "duration_model",
        "director_count",
        "cast_count",
        "country_count",
        "content_age",
        "is_multicountry",
        "is_long_content"
    ]

    for col in numeric_columns:

        model_df[col] = (
            model_df[col]
            .fillna(model_df[col].median())
        )


    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X = model_df.drop(
        columns=[target]
    )

    y = model_df[target]


    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    return model, encoders


tree_model, encoders = train_model(df)


# ============================================================
# FUNCIONES PARA CODIFICAR
# ============================================================

def encode_genre(value):

    return encoders["main_genre"].transform(
        [value]
    )[0]


def encode_rating(value):

    return encoders["rating"].transform(
        [value]
    )[0]


# ============================================================
# TÍTULO
# ============================================================

st.title(
    "🎬 Netflix Content Classifier"
)


# ============================================================
# INTRODUCCIÓN
# ============================================================

st.markdown(
    """
    ## ¿Para qué sirve este modelo?

    Este proyecto utiliza **Machine Learning** para clasificar
    automáticamente contenido de Netflix.

    El modelo analiza diferentes características del contenido,
    entre ellas:

    - 📅 Año de lanzamiento
    - ⏱️ Duración
    - 🎭 Género
    - 🔞 Clasificación
    - 🎥 Cantidad de directores
    - 🎭 Cantidad de actores
    - 🌎 Países involucrados
    - 📆 Antigüedad del contenido

    Con estas características, el modelo estima si el contenido
    corresponde a:

    **🎬 Movie** o **📺 TV Show**
    """
)


st.divider()


# ============================================================
# EJEMPLOS RÁPIDOS
# ============================================================

st.header(
    "🚀 Prueba rápida"
)

st.write(
    "Puedes cargar un ejemplo o introducir tus propias características."
)


col1, col2 = st.columns(2)


# ============================================================
# EJEMPLO PELÍCULA
# ============================================================

with col1:

    if st.button(
        "🎬 Cargar ejemplo de película",
        use_container_width=True
    ):

        st.session_state["content_input"] = "Película"

        st.session_state["release_year"] = 2020

        st.session_state["year_added"] = 2021

        st.session_state["month"] = "Junio"

        st.session_state["duration"] = 120

        st.session_state["directors"] = "1"

        st.session_state["cast"] = "6-10"

        st.session_state["countries"] = "1"

        st.session_state["age"] = 1

        st.session_state["multicountry"] = "No"

        st.session_state["long_content"] = "Sí"


        genres = list(
            encoders["main_genre"].classes_
        )

        if "Drama" in genres:

            st.session_state["genre"] = "Drama"

        else:

            st.session_state["genre"] = genres[0]


        ratings = list(
            encoders["rating"].classes_
        )

        if "PG-13" in ratings:

            st.session_state["rating"] = "PG-13"

        else:

            st.session_state["rating"] = ratings[0]


# ============================================================
# EJEMPLO SERIE
# ============================================================

with col2:

    if st.button(
        "📺 Cargar ejemplo de serie",
        use_container_width=True
    ):

        st.session_state["content_input"] = "Serie"

        st.session_state["release_year"] = 2019

        st.session_state["year_added"] = 2020

        st.session_state["month"] = "Junio"

        # 2 temporadas
        st.session_state["duration"] = 2

        st.session_state["directors"] = "1"

        st.session_state["cast"] = "6-10"

        st.session_state["countries"] = "1"

        st.session_state["age"] = 1

        st.session_state["multicountry"] = "No"

        st.session_state["long_content"] = "No"


        genres = list(
            encoders["main_genre"].classes_
        )

        if "Drama" in genres:

            st.session_state["genre"] = "Drama"

        else:

            st.session_state["genre"] = genres[0]


        ratings = list(
            encoders["rating"].classes_
        )

        if "TV-14" in ratings:

            st.session_state["rating"] = "TV-14"

        else:

            st.session_state["rating"] = ratings[0]


# ============================================================
# VALORES INICIALES
# ============================================================

if "content_input" not in st.session_state:

    st.session_state["content_input"] = "Película"


if "release_year" not in st.session_state:

    st.session_state["release_year"] = 2020


if "year_added" not in st.session_state:

    st.session_state["year_added"] = 2021


if "month" not in st.session_state:

    st.session_state["month"] = "Junio"


if "duration" not in st.session_state:

    st.session_state["duration"] = 90


if "directors" not in st.session_state:

    st.session_state["directors"] = "1"


if "cast" not in st.session_state:

    st.session_state["cast"] = "1-5"


if "countries" not in st.session_state:

    st.session_state["countries"] = "1"


if "age" not in st.session_state:

    st.session_state["age"] = 1


if "multicountry" not in st.session_state:

    st.session_state["multicountry"] = "No"


if "long_content" not in st.session_state:

    st.session_state["long_content"] = "No"


if "genre" not in st.session_state:

    st.session_state["genre"] = (
        encoders["main_genre"].classes_[0]
    )


if "rating" not in st.session_state:

    st.session_state["rating"] = (
        encoders["rating"].classes_[0]
    )


# ============================================================
# INFORMACIÓN BÁSICA
# ============================================================

st.header(
    "🔮 Realizar una predicción"
)

st.write(
    "Introduce las características del contenido."
)


st.subheader(
    "📅 Información básica"
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# AÑO LANZAMIENTO
# ------------------------------------------------------------

with col1:

    release_year = st.number_input(
        "Año de lanzamiento",
        min_value=1900,
        max_value=2030,
        step=1,
        key="release_year"
    )


# ------------------------------------------------------------
# AÑO AGREGADO
# ------------------------------------------------------------

with col2:

    year_added = st.number_input(
        "Año agregado a Netflix",
        min_value=2000,
        max_value=2030,
        step=1,
        key="year_added"
    )


# ------------------------------------------------------------
# MES
# ------------------------------------------------------------

with col3:

    months = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre"
    ]

    selected_month = st.selectbox(
        "Mes agregado",
        months,
        key="month"
    )


month_number = (
    months.index(selected_month) + 1
)


# ============================================================
# TIPO DE DURACIÓN
# ============================================================

st.subheader(
    "⏱️ Duración"
)


duration_input_type = st.radio(
    "¿Qué tipo de contenido estás describiendo?",
    [
        "Película",
        "Serie"
    ],
    horizontal=True,
    key="content_input"
)


# ============================================================
# DURACIÓN DE PELÍCULA
# ============================================================

if duration_input_type == "Película":

    duration_num = st.slider(
        "Duración de la película",
        min_value=30,
        max_value=300,
        value=min(
            max(
                st.session_state["duration"],
                30
            ),
            300
        ),
        step=5,
        key="movie_duration"
    )


    # Las películas ya vienen en minutos.
    duration_model = duration_num


    st.caption(
        f"🎬 Duración seleccionada: "
        f"**{duration_num} minutos**"
    )


# ============================================================
# DURACIÓN DE SERIE
# ============================================================

else:

    duration_num = st.slider(
        "Número de temporadas",
        min_value=1,
        max_value=15,
        value=min(
            max(
                st.session_state["duration"],
                1
            ),
            15
        ),
        step=1,
        key="series_duration"
    )


    # Convertimos temporadas a una escala comparable
    # con las películas.
    duration_model = duration_num * 30


    st.caption(
        f"📺 Temporadas seleccionadas: "
        f"**{duration_num}**"
    )


    st.caption(
        f"🔄 Valor utilizado por el modelo: "
        f"**{duration_model}**"
    )


# ============================================================
# CARACTERÍSTICAS
# ============================================================

st.subheader(
    "🎭 Características"
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# DIRECTORES Y ACTORES
# ------------------------------------------------------------

with col1:

    director_options = [
        "0",
        "1",
        "2",
        "3",
        "4 o más"
    ]


    directors = st.selectbox(
        "🎥 Número de directores",
        director_options,
        key="directors"
    )


    if directors == "4 o más":

        director_count = 4

    else:

        director_count = int(
            directors
        )


    cast_options = [
        "0",
        "1-5",
        "6-10",
        "11-20",
        "Más de 20"
    ]


    cast = st.selectbox(
        "🎭 Cantidad de actores",
        cast_options,
        key="cast"
    )


    cast_count_values = {

        "0": 0,

        "1-5": 3,

        "6-10": 8,

        "11-20": 15,

        "Más de 20": 25
    }


    cast_count = cast_count_values[
        cast
    ]


# ------------------------------------------------------------
# PAÍSES Y ANTIGÜEDAD
# ------------------------------------------------------------

with col2:

    country_options = [
        "1",
        "2",
        "3",
        "4 o más"
    ]


    countries = st.selectbox(
        "🌎 Países involucrados",
        country_options,
        key="countries"
    )


    country_count_values = {

        "1": 1,

        "2": 2,

        "3": 3,

        "4 o más": 4
    }


    country_count = country_count_values[
        countries
    ]


    age = st.slider(
        "📆 Antigüedad del contenido",
        min_value=0,
        max_value=20,
        value=st.session_state["age"],
        step=1,
        key="age"
    )


# ============================================================
# PRODUCCIÓN
# ============================================================

st.subheader(
    "🌎 Producción"
)


col1, col2 = st.columns(2)


with col1:

    multicountry = st.selectbox(
        "¿Participan varios países?",
        ["No", "Sí"],
        key="multicountry"
    )


    multicountry_value = (
        1
        if multicountry == "Sí"
        else 0
    )


with col2:

    long_content = st.selectbox(
        "¿Consideras que es contenido largo?",
        ["No", "Sí"],
        key="long_content"
    )


    long_content_value = (
        1
        if long_content == "Sí"
        else 0
    )


# ============================================================
# GÉNERO Y CLASIFICACIÓN
# ============================================================

st.subheader(
    "🏷️ Género y clasificación"
)


col1, col2 = st.columns(2)


with col1:

    main_genre = st.selectbox(
        "🎭 Género principal",
        encoders["main_genre"].classes_,
        key="genre"
    )


with col2:

    rating = st.selectbox(
        "🔞 Clasificación",
        encoders["rating"].classes_,
        key="rating"
    )


# ============================================================
# BOTÓN
# ============================================================

st.divider()


col1, col2, col3 = st.columns(
    [1, 2, 1]
)


with col2:

    predict = st.button(
        "🔮 CLASIFICAR CONTENIDO",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICCIÓN
# ============================================================

if predict:

    genre_encoded = encode_genre(
        main_genre
    )


    rating_encoded = encode_rating(
        rating
    )


    # --------------------------------------------------------
    # DATOS PARA EL MODELO
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "release_year": release_year,

        "year_added": year_added,

        "month_added": month_number,

        "duration_model": duration_model,

        "director_count": director_count,

        "cast_count": cast_count,

        "country_count": country_count,

        "content_age": age,

        "is_multicountry": multicountry_value,

        "is_long_content": long_content_value,

        "main_genre": genre_encoded,

        "rating": rating_encoded

    }])


    # --------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------

    prediction = tree_model.predict(
        input_data
    )


    predicted_class = (
        encoders["type"]
        .inverse_transform(
            prediction
        )[0]
    )


    # --------------------------------------------------------
    # PROBABILIDADES
    # --------------------------------------------------------

    probabilities = (
        tree_model
        .predict_proba(
            input_data
        )[0]
    )


    classes = (
        encoders["type"]
        .classes_
    )


    confidence = max(
        probabilities
    )


    # ========================================================
    # RESULTADO
    # ========================================================

    st.divider()

    st.header(
        "🎯 Resultado"
    )


    # --------------------------------------------------------
    # PELÍCULA
    # --------------------------------------------------------

    if predicted_class == "Movie":

        st.success(
            "🎬 El modelo predice que este contenido "
            "es una **PELÍCULA**."
        )


        image_url = (
            "https://images.unsplash.com/"
            "photo-1489599849927-2ee91cede3ba"
            "?auto=format&fit=crop&w=1200&q=80"
        )


        col1, col2 = st.columns(
            [1.5, 1]
        )


        with col1:

            st.image(
                image_url,
                caption="🎬 Movie",
                use_container_width=True
            )


        with col2:

            st.metric(
                "Resultado",
                "🎬 MOVIE"
            )


            st.metric(
                "Confianza",
                f"{confidence:.1%}"
            )


    # --------------------------------------------------------
    # SERIE
    # --------------------------------------------------------

    else:

        st.info(
            "📺 El modelo predice que este contenido "
            "es una **SERIE DE TELEVISIÓN**."
        )


        image_url = (
            "https://images.unsplash.com/"
            "photo-1522869635100-9f4c5e86aa37"
            "?auto=format&fit=crop&w=1200&q=80"
        )


        col1, col2 = st.columns(
            [1.5, 1]
        )


        with col1:

            st.image(
                image_url,
                caption="📺 TV Show",
                use_container_width=True
            )


        with col2:

            st.metric(
                "Resultado",
                "📺 TV SHOW"
            )


            st.metric(
                "Confianza",
                f"{confidence:.1%}"
            )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    st.subheader(
        "📊 Probabilidad de clasificación"
    )


    probability_df = pd.DataFrame({

        "Tipo de contenido": classes,

        "Probabilidad": probabilities * 100

    })


    st.bar_chart(
        probability_df.set_index(
            "Tipo de contenido"
        )
    )


    # ========================================================
    # INFORMACIÓN UTILIZADA
    # ========================================================

    with st.expander(
        "🔎 Ver información utilizada por el modelo"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Proyecto de Machine Learning — "
    "Netflix Content Classification"
)

