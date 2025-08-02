## Análisis de Rendimiento Estudiantil

Este proyecto realiza un análisis avanzado del rendimiento académico de estudiantes, utilizando 
herramientas de estadística, visualización de datos y aprendizaje automático. El objetivo es descubrir patrones y relaciones relevantes entre variables como género, nivel educativo de los padres y puntajes en distintas materias.

## Estructura General del Proyecto

El análisis se organiza en varias etapas, cada una abordando aspectos fundamentales 
del procesamiento y entendimiento de los datos:

1. Carga y preprocesamiento de datos
2. Análisis exploratorio
3. Visualización de datos
4. Machine Learning
5. Análisis estadístico avanzado

## Requisitos para ejecutar
Antes de ejecutar el script, asegúrate de tener instaladas las siguientes bibliotecas:

pip install pandas matplotlib seaborn scikit-learn scipy

## Archivo de entrada
StudentsPerformance.csv: archivo que contiene los datos de estudiantes, incluyendo género, nivel educativo de los padres y puntuaciones en matemáticas, lectura y escritura.
Archivo obtenido de kaggle "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams"

## Ejecución
Para correr el análisis completo, simplemente ejecuta:

python trabajo3.py

El script realiza todos los pasos automáticamente e imprime resúmenes y resultados en la consola.

## Descripción de funciones clave

statistical_analysis(df)
Realiza análisis estadísticos avanzados, incluyendo:

Test t de Student para comparar los puntajes de matemáticas entre hombres y mujeres.
ANOVA para evaluar diferencias en puntajes promedio según el nivel educativo de los padres.
Correlaciones de Pearson entre las distintas materias (matemáticas, lectura, escritura), junto con sus valores p.

Esta función utiliza la biblioteca scipy.stats. Si no está instalada, el análisis estadístico será omitido con una advertencia.

main()
Función principal que orquesta todo el análisis:

* Carga y preprocesa los datos.
* Ejecuta análisis exploratorio.
* Genera visualizaciones.
* Aplica modelos de aprendizaje automático si scikit-learn está disponible.
* Llama al análisis estadístico avanzado.
* Al final, imprime un resumen con métricas clave del análisis.

## Resultados Esperados
Visualizaciones generadas: 18 gráficos.
Modelo de ML: Entrenado exitosamente (si sklearn está disponible).
Tests estadísticos realizados: 3 pruebas (Test T, ANOVA, Correlaciones).

## Posibles errores
Si el archivo StudentsPerformance.csv no se encuentra, el script mostrará un mensaje de error claro.

Si alguna biblioteca no está instalada, se omitirán funciones específicas (como scipy o sklearn).

## Notas adicionales
Este análisis es ideal para fines educativos, especialmente para quienes están aprendiendo sobre ciencia de datos aplicada al análisis educativo.

## Autores
Juan Miguel Llumihuasi