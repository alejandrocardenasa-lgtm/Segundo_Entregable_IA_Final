import streamlit as st
import joblib
import pandas as pd

from tensorflow import keras
from pathlib import Path


# ============================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Sistema de Alerta Temprana Saber 11",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# 2. RUTAS DEL PROYECTO
# ============================================================

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MODELOS = RUTA_PROYECTO / "modelos"

ARCHIVO_MODELO = (
    RUTA_MODELOS / "modelo_mlp_saber11_temporal_2025.keras"
)

ARCHIVO_PREPROCESADOR = (
    RUTA_MODELOS / "preprocesador_saber11_temporal_2025.pkl"
)

ARCHIVO_UMBRAL = (
    RUTA_MODELOS / "umbral_saber11_temporal_2025.pkl"
)


# ============================================================
# 3. CARGAR MODELO, PREPROCESADOR Y UMBRAL
# ============================================================

@st.cache_resource
def cargar_modelos():

    modelo = keras.models.load_model(
        ARCHIVO_MODELO,
        compile=False
    )

    preprocesador = joblib.load(
        ARCHIVO_PREPROCESADOR
    )

    umbral = joblib.load(
        ARCHIVO_UMBRAL
    )

    return modelo, preprocesador, float(umbral)


try:

    modelo, preprocesador, umbral = cargar_modelos()

except FileNotFoundError as error:

    st.error(
        "No se encontraron los archivos del modelo temporal."
    )

    st.write(
        "Verifica que los tres archivos estén "
        "guardados en la carpeta 'modelos'."
    )

    st.code(str(error))

    st.stop()

except Exception as error:

    st.error(
        "Ocurrió un error al cargar el modelo."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 4. VARIABLES UTILIZADAS POR EL MODELO
# ============================================================

variables_modelo = [
    "cole_jornada",
    "cole_calendario",
    "cole_naturaleza",
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_caracter",
    "fami_tieneinternet",
    "fami_tienecomputador",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_estratovivienda",
    "estu_genero",
    "grupo_edad"
]


# ============================================================
# 5. OBTENER LAS CATEGORÍAS REALES DEL PREPROCESADOR
# ============================================================

def obtener_categorias_preprocesador(preprocesador):

    categorias = {}

    for nombre, transformador, columnas in preprocesador.transformers_:

        if nombre == "remainder" or transformador == "drop":
            continue

        encoder = transformador

        # Si el transformador es un Pipeline,
        # buscamos el OneHotEncoder dentro de él.
        if hasattr(transformador, "named_steps"):

            for paso in transformador.named_steps.values():

                if hasattr(paso, "categories_"):
                    encoder = paso
                    break

        if hasattr(encoder, "categories_"):

            for columna, valores in zip(
                columnas,
                encoder.categories_
            ):

                categorias[columna] = [
                    str(valor) for valor in valores
                ]

    return categorias


categorias_modelo = obtener_categorias_preprocesador(
    preprocesador
)


# ============================================================
# 6. FUNCIÓN PARA MOSTRAR LAS OPCIONES DEL FORMULARIO
# ============================================================

def seleccionar_categoria(etiqueta, columna):

    opciones = categorias_modelo.get(columna, [])

    if not opciones:

        st.error(
            f"No se encontraron categorías para: {columna}"
        )

        st.stop()

    # Si existe "No reportado", se utiliza como
    # opción inicial para no asumir información.
    if "No reportado" in opciones:

        indice_inicial = opciones.index("No reportado")

    else:

        indice_inicial = None

    return st.selectbox(
        etiqueta,
        options=opciones,
        index=indice_inicial,
        placeholder="Selecciona una opción",
        key=columna
    )


# ============================================================
# 7. FUNCIÓN PARA CLASIFICAR LA EDAD
# ============================================================

def obtener_grupo_edad(edad):

    if edad is None:
        return "No reportado"

    if edad <= 15:
        return "15 o menos"

    elif edad == 16:
        return "16"

    elif edad == 17:
        return "17"

    elif edad == 18:
        return "18"

    elif edad == 19:
        return "19"

    elif 20 <= edad <= 24:
        return "20 a 24"

    elif 25 <= edad <= 29:
        return "25 a 29"

    else:
        return "30 o más"


# ============================================================
# 8. PRESENTACIÓN DE LA APLICACIÓN
# ============================================================

st.title("🎓 Sistema de Alerta Temprana - Saber 11")

st.write(
    "Esta aplicación utiliza un modelo de inteligencia "
    "artificial para estimar el riesgo de obtener un "
    "puntaje global inferior a 250 en las pruebas "
    "Saber 11, a partir de características de "
    "estudiantes de Cali."
)

st.write(
    "El objetivo es apoyar la identificación de "
    "posibles situaciones de riesgo académico y "
    "orientar estrategias de acompañamiento educativo."
)

st.info(
    "El sistema utiliza una red neuronal MLP entrenada "
    "con registros históricos anteriores a 2025 y "
    "evaluada con datos de ese año."
)

st.divider()

st.subheader("📋 Evaluación del estudiante")

st.write(
    "Completa la información del estudiante, "
    "su colegio y su entorno familiar."
)


# ============================================================
# 9. FORMULARIO DEL ESTUDIANTE
# ============================================================

with st.form("formulario_estudiante"):

    # --------------------------------------------------------
    # INFORMACIÓN DEL COLEGIO
    # --------------------------------------------------------

    st.markdown("### 🏫 Información del colegio")

    jornada = seleccionar_categoria(
        "Jornada escolar",
        "cole_jornada"
    )

    calendario = seleccionar_categoria(
        "Calendario escolar",
        "cole_calendario"
    )

    naturaleza = seleccionar_categoria(
        "Naturaleza del colegio",
        "cole_naturaleza"
    )

    area = seleccionar_categoria(
        "Ubicación del colegio",
        "cole_area_ubicacion"
    )

    bilingue = seleccionar_categoria(
        "¿El colegio es bilingüe?",
        "cole_bilingue"
    )

    caracter = seleccionar_categoria(
        "Carácter del colegio",
        "cole_caracter"
    )

    # --------------------------------------------------------
    # INFORMACIÓN FAMILIAR
    # --------------------------------------------------------

    st.markdown("### 🏠 Información familiar")

    internet = seleccionar_categoria(
        "¿Tiene internet en casa?",
        "fami_tieneinternet"
    )

    computador = seleccionar_categoria(
        "¿Tiene computador en casa?",
        "fami_tienecomputador"
    )

    educacion_madre = seleccionar_categoria(
        "Nivel educativo de la madre",
        "fami_educacionmadre"
    )

    educacion_padre = seleccionar_categoria(
        "Nivel educativo del padre",
        "fami_educacionpadre"
    )

    estrato = seleccionar_categoria(
        "Estrato de la vivienda",
        "fami_estratovivienda"
    )

    # --------------------------------------------------------
    # INFORMACIÓN DEL ESTUDIANTE
    # --------------------------------------------------------

    st.markdown("### 👤 Información del estudiante")

    genero = seleccionar_categoria(
        "Género registrado",
        "estu_genero"
    )

    edad_no_reportada = st.checkbox(
        "No se conoce la edad del estudiante"
    )

    if edad_no_reportada:

        edad = None

        st.caption(
            "Se utilizará la categoría 'No reportado' "
            "para la edad."
        )

    else:

        edad = st.number_input(
            "Edad del estudiante",
            min_value=10,
            max_value=100,
            value=16,
            step=1
        )

    st.divider()

    evaluar = st.form_submit_button(
        "🔍 Evaluar riesgo académico",
        use_container_width=True
    )


# ============================================================
# 10. REALIZAR LA PREDICCIÓN
# ============================================================

if evaluar:

    # --------------------------------------------------------
    # VALIDAR CAMPOS OBLIGATORIOS
    # --------------------------------------------------------

    respuestas = {
        "Jornada escolar": jornada,
        "Calendario escolar": calendario,
        "Naturaleza del colegio": naturaleza,
        "Ubicación del colegio": area,
        "Colegio bilingüe": bilingue,
        "Carácter del colegio": caracter,
        "Internet": internet,
        "Computador": computador,
        "Educación de la madre": educacion_madre,
        "Educación del padre": educacion_padre,
        "Estrato": estrato,
        "Género": genero
    }

    campos_sin_seleccionar = [
        nombre
        for nombre, valor in respuestas.items()
        if valor is None
    ]

    if campos_sin_seleccionar:

        st.error(
            "Completa los campos pendientes antes "
            "de realizar la evaluación."
        )

        st.write(
            "Campos pendientes: "
            + ", ".join(campos_sin_seleccionar)
        )

        st.stop()

    # --------------------------------------------------------
    # CONVERTIR LA EDAD A SU GRUPO
    # --------------------------------------------------------

    grupo_edad = obtener_grupo_edad(edad)

    # --------------------------------------------------------
    # CONSTRUIR EL REGISTRO
    # --------------------------------------------------------

    estudiante = pd.DataFrame([{

        "cole_jornada": jornada,
        "cole_calendario": calendario,
        "cole_naturaleza": naturaleza,
        "cole_area_ubicacion": area,
        "cole_bilingue": bilingue,
        "cole_caracter": caracter,

        "fami_tieneinternet": internet,
        "fami_tienecomputador": computador,
        "fami_educacionmadre": educacion_madre,
        "fami_educacionpadre": educacion_padre,
        "fami_estratovivienda": estrato,

        "estu_genero": genero,
        "grupo_edad": grupo_edad

    }])

    # Mantener el mismo orden de columnas
    estudiante = estudiante[variables_modelo]

    # --------------------------------------------------------
    # MOSTRAR LOS DATOS INGRESADOS
    # --------------------------------------------------------

    with st.expander(
        "🔎 Verificar información ingresada"
    ):

        st.dataframe(
            estudiante.T,
            use_container_width=True
        )

    # --------------------------------------------------------
    # VERIFICAR CATEGORÍAS
    # --------------------------------------------------------

    categorias_desconocidas = []

    for columna in variables_modelo:

        opciones_validas = categorias_modelo.get(
            columna,
            []
        )

        valor = estudiante[columna].iloc[0]

        if valor not in opciones_validas:

            categorias_desconocidas.append(
                f"{columna}: {valor}"
            )

    if categorias_desconocidas:

        st.error(
            "Algunos valores no coinciden con "
            "las categorías del modelo."
        )

        for categoria in categorias_desconocidas:

            st.write(f"- {categoria}")

        st.stop()

    # --------------------------------------------------------
    # PREPROCESAMIENTO Y PREDICCIÓN
    # --------------------------------------------------------

    try:

        estudiante_procesado = preprocesador.transform(
            estudiante
        )

        salida_modelo = modelo.predict(
            estudiante_procesado,
            verbose=0
        )

        puntuacion = float(
            salida_modelo[0][0]
        )

        prediccion = int(
            puntuacion >= umbral
        )

    except Exception as error:

        st.error(
            "Ocurrió un problema al procesar "
            "la información del estudiante."
        )

        st.exception(error)

        st.stop()

    # ========================================================
    # 11. MOSTRAR RESULTADOS
    # ========================================================

    st.divider()

    st.subheader("📊 Resultado de la evaluación")

    st.metric(
        label="Puntuación estimada de bajo desempeño",
        value=f"{puntuacion * 100:.2f}%"
    )

    st.progress(
        min(max(puntuacion, 0.0), 1.0)
    )

    st.caption(
        "Esta puntuación es la salida del modelo MLP. "
        "No debe interpretarse como una probabilidad "
        "individual calibrada."
    )

    st.write(
        f"**Umbral de clasificación:** {umbral:.2f}"
    )

    # --------------------------------------------------------
    # RESULTADO: POSIBLE BAJO DESEMPEÑO
    # --------------------------------------------------------

    if prediccion == 1:

        st.warning(
            "🟠 ALERTA: Posible bajo desempeño académico"
        )

        st.write(
            "El modelo identificó características "
            "asociadas con un posible puntaje global "
            "inferior a 250 en Saber 11."
        )

        st.write(
            "Este resultado sugiere que podría ser "
            "conveniente realizar una evaluación "
            "académica adicional."
        )

        st.markdown(
            "### 📚 Recomendaciones académicas"
        )

        st.write(
            "• Realizar una evaluación diagnóstica "
            "para identificar posibles dificultades."
        )

        st.write(
            "• Implementar actividades de refuerzo "
            "en las áreas que lo requieran."
        )

        st.write(
            "• Hacer seguimiento periódico "
            "al progreso académico."
        )

        st.write(
            "• Promover el acompañamiento "
            "de docentes y familiares."
        )

    # --------------------------------------------------------
    # RESULTADO: SIN ALERTA
    # --------------------------------------------------------

    else:

        st.success(
            "🟢 No se identificó una alerta de bajo desempeño"
        )

        st.write(
            "La puntuación del modelo se encuentra "
            "por debajo del umbral utilizado para "
            "identificar posibles casos de bajo desempeño."
        )

        st.write(
            "Esto no garantiza que el estudiante "
            "obtendrá un resultado satisfactorio."
        )

        st.markdown(
            "### 📚 Recomendaciones académicas"
        )

        st.write(
            "• Mantener los hábitos de estudio."
        )

        st.write(
            "• Continuar con el seguimiento académico."
        )

        st.write(
            "• Reforzar las áreas en las que "
            "se identifiquen dificultades."
        )

    # ========================================================
    # 12. INFORMACIÓN DEL MODELO
    # ========================================================

    st.divider()

    st.markdown("### 🧠 Información del modelo")

    st.write(
        "**Modelo:** Red neuronal artificial "
        "Multilayer Perceptron (MLP)"
    )

    st.write(
        "**Tipo de aprendizaje:** Supervisado"
    )

    st.write(
        "**Problema:** Clasificación binaria"
    )

    st.write(
        "**Variable objetivo:** Puntaje global "
        "inferior a 250 en Saber 11"
    )

    st.write(
        "**Población estudiada:** "
        "Estudiantes de colegios ubicados en Cali"
    )

    st.write(
        "**Dataset:** 128.260 registros históricos "
        "de 2019, 2020 y 2022–2025"
    )

    st.write(
        "**Entrenamiento y validación:** "
        "Registros de 2019 a 2024"
    )

    st.write(
        "**Evaluación temporal:** "
        "20.764 registros de 2025"
    )

    st.write(
        "**Variables predictoras:** 13"
    )

    st.markdown(
        "### 📈 Resultados de la evaluación temporal"
    )

    st.write(
        "**Exactitud:** 63,16 %"
    )

    st.write(
        "**Precisión:** 52,58 %"
    )

    st.write(
        "**Sensibilidad:** 84,87 %"
    )

    st.write(
        "**F1-score:** 64,93 %"
    )

    st.write(
        "**AUC:** 0,7532"
    )

    st.info(
        "Estos resultados corresponden a la evaluación "
        "del modelo temporal con registros de 2025. "
        "El sistema también puede generar falsas alertas "
        "y dejar de identificar algunos casos."
    )

    st.warning(
        "Esta herramienta es un prototipo académico. "
        "Sus resultados no deben utilizarse como "
        "diagnóstico definitivo ni como única base "
        "para tomar decisiones sobre estudiantes. "
        "Tampoco se ha demostrado que permita anticipar "
        "los resultados del examen con varios años "
        "de anticipación."
    )


# ============================================================
# 13. PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Proyecto académico de Inteligencia Artificial | "
    "Predicción de bajo desempeño en Saber 11 | "
    "Cali, Colombia | Datos históricos 2019–2025"
)