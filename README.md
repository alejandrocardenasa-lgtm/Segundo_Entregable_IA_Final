# Sistema de Alerta Temprana — Saber 11 Cali

Proyecto académico de Inteligencia Artificial. Clasificación de puntaje global inferior a 250 puntos.

## Contenido de la entrega

- `scripts/02_Modelos_Predictivos_Saber11_Cali_ORGANIZADO_V3.ipynb`: notebook principal. Incluye comparación de modelos, entrenamiento MLP, selección del umbral en validación y prueba temporal con registros de 2025.
- `data/processed/saber11_cali_2019_2025_limpio.csv`: dataset limpio para ejecutar el notebook principal (128.260 registros).
- `app/app.py`: aplicación Streamlit.
- `modelos/`: los tres archivos del modelo temporal que carga la aplicación.
- `documentacion/`: documento del proyecto.
- `EDA_Saber11_Cali_EDA_TODOS_ANOS_SIN_MODELO.ipynb`: análisis exploratorio complementario histórico.
- `data/raw/saber11_cali_2019_2022.csv`: archivo de entrada para el EDA complementario.
- `figs/`: figuras de ese EDA.

## Ejecución

1. Descomprimir el ZIP y abrir `Primer_Entregable_IA` en VS Code.
2. Crear un entorno virtual e instalar las dependencias con `pip install -r requirements.txt`.
3. Abrir `scripts/02_Modelos_Predictivos_Saber11_Cali_ORGANIZADO_V3.ipynb` y ejecutar las celdas en orden, con el directorio de trabajo configurado en `scripts/` (rutas relativas a `../data` y `../modelos`).
4. Para iniciar la app, desde la carpeta principal ejecutar `streamlit run app/app.py`. La aplicación carga los archivos guardados; no requiere reentrenar el modelo.
5. El EDA complementario se ejecuta desde la carpeta principal porque usa `data/raw/` y `figs/`.

## Alcance y advertencias

El modelo usa 13 variables, se entrena y valida con registros de 2019–2024 y se evalúa en 2025. El umbral es 0,35. La sensibilidad observada en 2025 es 84,87 % y la precisión 52,58 %. Es un prototipo académico: no constituye un diagnóstico ni una probabilidad individual calibrada.

El CSV procesado es el insumo principal para reproducir el entrenamiento y los resultados. Los archivos crudos masivos de origen no se adjuntan porque no son necesarios para ejecutar el notebook final; se conserva un archivo crudo filtrado para el EDA complementario. La reconstrucción integral del CSV procesado a partir de todas las fuentes originales no está incluida en este paquete.
