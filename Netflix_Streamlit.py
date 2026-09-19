
import pandas as pd
import streamlit as st

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
# CARGAR DATOS
# ============================================================

URL_DATASET = (
    "https://raw.githubusercontent.com/"
    "JoseColins1OO/Proyecto_Netflix/"
    "refs/heads/main/data/processed/netflix_clean.csv"
)


@st.cache_data
def cargar_datos():

    df = pd.read_csv(URL_DATASET)

    df["duration_unit"] = (
        df["duration_unit"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    def normalizar_duracion(row):

        if pd.isna(row["duration_num"]):
            return None

        if row["duration_unit"] in [
            "min",
            "mins",
            "minute",
            "minutes"
        ]:
            return row["duration_num"]

        if row["duration_unit"] in [
            "season",
            "seasons"
        ]:
            return row["duration_num"] * 30

        return row["duration_num"]

    df["duration_model"] = df.apply(
        normalizar_duracion,
        axis=1
    )

    return df


df = cargar_datos()


# ============================================================
# ENTRENAMIENTO
# ============================================================

@st.cache_resource
def entrenar(df):

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

    model_df = df[
        features + ["type"]
    ].copy()

    encoders = {}

    for column in [
        "main_genre",
        "rating",
        "type"
    ]:

        encoder = LabelEncoder()

        model_df[column] = encoder.fit_transform(
            model_df[column].astype(str)
        )

        encoders[column] = encoder

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

    for column in numeric_columns:

        model_df[column] = model_df[column].fillna(
            model_df[column].median()
        )

    X = model_df.drop(
        columns=["type"]
    )

    y = model_df["type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    modelo = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )

    modelo.fit(
        X_train,
        y_train
    )

    return modelo, encoders


modelo, encoders = entrenar(df)



# ============================================================
# ENCABEZADO
# ============================================================

st.title("🎬 Netflix Content Classifier")

st.markdown("""
### Integrantes del equipo

**José Manuel Colín Aguilar**  
**Ana Nicole Martinez Morales**  
**Diego Liberato Jury**
""")


# ============================================================
# CONTEXTO DEL PROYECTO
# ============================================================

st.divider()

st.markdown("""
## Contexto del proyecto

Netflix cuenta con un amplio catálogo de contenidos que incluye
**películas y series** con diferentes características de producción,
clasificación, género, duración y distribución.

En este proyecto se desarrolló un modelo de **Machine Learning**
utilizando un **árbol de decisión**, cuyo objetivo es analizar las
características de un contenido y determinar automáticamente si
corresponde a una **película** o una **serie**.

### ¿Cómo funciona?

El usuario proporciona algunas características del contenido,
como:

- 🎞️ Año de lanzamiento
- 📅 Año de incorporación a Netflix
- ⏱️ Duración aproximada
- 🎭 Género
- 🔞 Clasificación
- 🎬 Número de directores y actores
- 🌎 Número de países involucrados
- 📊 Otras características del contenido

A partir de estos datos, el modelo analiza los patrones aprendidos
del catálogo de Netflix y genera una **predicción**, acompañada de
la probabilidad estimada para cada tipo de contenido.

> **El reto:** proporciona las características del contenido,
> pero no indiques si es una película o una serie.
> **¡Deja que el modelo lo descubra!**
""")


# ============================================================
# SEPARADOR
# ============================================================

st.divider()

st.markdown("""
## 🔮 Realiza una predicción

Ingresa las características del contenido y presiona
**"Clasificar contenido"** para conocer el resultado.
""")


# ============================================================
# ENTRADA
# ============================================================

col1, col2 = st.columns(2)


with col1:

    release_year = st.number_input(
        "Año de lanzamiento",
        1920,
        2026,
        2020
    )

    year_added = st.number_input(
        "Año de incorporación",
        2008,
        2026,
        2021
    )

    month_added = st.slider(
        "Mes de incorporación",
        1,
        12,
        6
    )

    duration = st.slider(
        "Duración aproximada",
        1,
        300,
        120
    )


with col2:

    director_count = st.number_input(
        "Número de directores",
        0,
        20,
        1
    )

    cast_count = st.number_input(
        "Número de actores",
        0,
        50,
        10
    )

    country_count = st.number_input(
        "Número de países",
        1,
        20,
        1
    )

    content_age = st.number_input(
        "Antigüedad del contenido",
        0,
        30,
        1
    )


# ============================================================
# PRODUCCIÓN
# ============================================================

multicountry = st.selectbox(
    "¿Participan varios países?",
    ["No", "Sí"]
)

is_multicountry = (
    1 if multicountry == "Sí"
    else 0
)

long_content = st.selectbox(
    "¿Contenido largo?",
    ["No", "Sí"]
)

is_long_content = (
    1 if long_content == "Sí"
    else 0
)


# ============================================================
# CATEGORÍAS
# ============================================================

genre = st.selectbox(
    "Género",
    sorted(
        df["main_genre"]
        .dropna()
        .astype(str)
        .unique()
    )
)

rating = st.selectbox(
    "Clasificación",
    sorted(
        df["rating"]
        .dropna()
        .astype(str)
        .unique()
    )
)


# ============================================================
# PREDICCIÓN
# ============================================================

if st.button(
    "Clasificar contenido",
    use_container_width=True
):

    genre_encoded = encoders[
        "main_genre"
    ].transform([genre])[0]

    rating_encoded = encoders[
        "rating"
    ].transform([rating])[0]

    entrada = pd.DataFrame([
        {
            "release_year": release_year,
            "year_added": year_added,
            "month_added": month_added,
            "duration_model": duration,
            "director_count": director_count,
            "cast_count": cast_count,
            "country_count": country_count,
            "content_age": content_age,
            "is_multicountry": is_multicountry,
            "is_long_content": is_long_content,
            "main_genre": genre_encoded,
            "rating": rating_encoded
        }
    ])

    prediccion = modelo.predict(entrada)

    resultado = encoders[
        "type"
    ].inverse_transform(
        prediccion
    )[0]

    probabilidades = modelo.predict_proba(
        entrada
    )[0]

    clases = encoders[
        "type"
    ].classes_


    # ========================================================
    # RESULTADO
    # ========================================================

    st.subheader("🔮 Resultado")

    if resultado == "Movie":

        st.success(
            "🎬 El modelo predice: **PELÍCULA**"
        )

        st.image(
            "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba"
            "?auto=format&fit=crop&w=1200&q=80",
            caption="🎬 Película",
            use_container_width=True
        )

    else:

        st.info(
            "📺 El modelo predice: **SERIE**"
        )

        st.image(
            "https://images.unsplash.com/photo-1593784991095-a205069470b6"
            "?auto=format&fit=crop&w=1200&q=80",
            caption="📺 Serie",
            use_container_width=True
        )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    st.subheader("📊 Probabilidades")

    resultados = pd.DataFrame({
        "Tipo": clases,
        "Probabilidad": probabilidades * 100
    })

    resultados["Probabilidad"] = (
        resultados["Probabilidad"]
        .round(2)
    )

    st.bar_chart(
        resultados.set_index("Tipo")
    )


    # ========================================================
    # CONFIANZA
    # ========================================================

    confianza = max(probabilidades) * 100

    st.metric(
        "Confianza",
        f"{confianza:.2f}%"
    )


    # ========================================================
    # DATOS
    # ========================================================

    with st.expander(
        "🔎 Ver datos utilizados"
    ):

        st.dataframe(
            entrada
        )

