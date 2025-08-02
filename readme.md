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

### Análisis y Conclusiones del Rendimiento Estudiantil

## Resumen

El análisis de 1,000 estudiantes revela patrones significativos en el rendimiento académico influenciados por factores demográficos, socioeconómicos y educativos.

## Hallazgos Principales

1. Rendimiento General (Tabla Estadistica Descriptiva)
- Promedio global: 67.77 puntos (sobre 100)
- Analisis del dato: El dato se puede encontrar en la tabala Estadisticas Descriptivas en la columna average_score de la fila mean. 
- Esto explica el rendimiento general que se obtiene de los resultados de los examenes matematicos, de lectura, y de escritura dado a todos los estudiantes.  


2. Disparidades de Género (Tabala Analisis pro Genero)
Patrón diferenciado por materia:
- Matemáticas: Los hombres superan a las mujeres por 5.1 puntos (68.73 vs 63.63)
- Lectura: Las mujeres superan a los hombres por 7.1 puntos (72.61 vs 65.47)
- Escritura: Las mujeres superan a los hombres por 9.2 puntos (72.47 vs 63.31)
- Analisis del dato: Estos datos vienen de la tabla Analisis por genero, en donde se puede observar la brecha de genero que refleja patrones de preferencias académicas

3. Impacto Socioeconómico Crítico (Tabala Analisis socioeconomico)
- Diferencia de 8.6 puntos entre estudiantes con almuerzo estándar vs gratuito/reducido
- Analisis del dato: En esta tabla podemos observar que los estudiantes que tienen el almuerzo standard obtuvieron mejores resultados, se puede deducir tambien que la mayoria de ellos estan en mejor situcion economica los cual les permite concentrarse mas en los estudios. 

4. Educación Parental como Factor Clave (Tabla Eduacion Parental)
- Analisis del dato: Podemos observar que los estudiantes que cuentan con padres con mayos grado de educacion obtienen mejores resultados en los examenes ya que existe una diferencia de 10.5 puntos entre los alumnos con padres que solo completaron la secundaria y padres con maestrias.

5. Efectividad de la Preparación Académica (Tabala Curso de Preparacion)
- Los estudiantes con preparación obtuvieron 72.67 vs 65.04 puntos de los que no tuvieron preparacion
- Analisis del dato: Con esta informacion podemos observar lo importante que es tomar un curso de preparacion para el examen, los estudiantes preparados obtivieron 7.6 puntos mas que los estudiantes que no se prepararon. 

## Análisis de Machine Learning

### Resultados del Modelo
- Analisis del dato: Existen factores no capturados en el dataset

### Importancia de Variables
1. Educación parental (28.9%): Factor más influyente
2. Etnia (26.8%): Segundo factor más relevante
3. Estatus socioeconómico (18.1%): Tercer factor
4. Preparación académica (15.2%): Cuarto factor
5. Género (11.1%): Menor impacto relativo

## Recomendaciones

### Para Políticas Educativas
1. Equidad socioeconómica: Implementar programas de apoyo para estudiantes de bajos recursos
2. Programas de preparación: Expandir el acceso a cursos de preparación académica
3. Apoyo familiar: Desarrollar programas de educación parental y participación familiar

### Para Instituciones Educativas
1. Pedagogía diferenciada: Implementar métodos específicos para abordar brechas de género

## Conclusión Final

El análisis revela un sistema educativo con disparidades sistemáticas basadas en factores socioeconómicos, familiares y demográficos. Aunque existen brechas significativas, la alta correlación entre diferentes habilidades académicas y el impacto positivo de la preparación académica sugieren oportunidades claras para intervenciones efectivas. 

La prioridad debe ser abordar las inequidades socioeconómicas mientras se desarrollan estrategias pedagógicas que aprovechen la interconexión de las habilidades académicas para maximizar el impacto educativo.

## Autores
Juan Miguel Llumihuasi
Israel Alvarez Reinoso
Luz Mollacano Sabando
