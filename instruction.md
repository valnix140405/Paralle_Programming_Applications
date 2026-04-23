Eres un experto en High Performance Computing (HPC), Data Engineering y Python avanzado. Tu objetivo es ayudarme a planificar e implementar la resolución de dos ejercicios de computación paralela basados en el archivo exercises.md adjunto.


El entorno de desarrollo utilizará mpi4py para la paralelización en memoria distribuida. IMPORTANTE: No generes ningún reporte, ensayo ni documentación final por el momento. Nuestro único objetivo ahora es lograr que el código funcione correctamente (versiones seriales y paralelas), fijar semillas aleatorias para reproducibilidad, y medir correctamente los tiempos de ejecución, speedup y eficiencia.

Project Structure Requirements:

Crea esta estructura de directorios antes de codificar:


/exercise_3/ (Código para el Autómata Celular y NASA FIRMS)

/exercise_4/ (Código para K-Means y Covertype dataset)

/docs/assets/ (Para guardar gráficas o logs de rendimiento)

requirements.txt (Para asegurar la reproducibilidad)

Execution Plan:

Procede paso a paso. Detente y pide mi confirmación al terminar cada fase antes de avanzar.

Fase 1: Preparación y Estrategia



Genera el requirements.txt con las dependencias exactas (mpi4py, numpy, pandas, requests para la API de FIRMS, etc.).

Explícame en viñetas cortas cómo implementarás la partición de datos:

Ex 3: Domain decomposition (intercambio de fronteras/ghost cells para el grid 2D).

Ex 4: Distribución de observaciones de Covertype y uso de Allreduce para los centroides.

Fase 2: Implementación - Exercise 3 (Forest Fire CA - NASA FIRMS)Contexto: Estados de celdas: 0 (no quemable), 1 (susceptible), 2 (quemándose), 3 (quemado).



Escribe un script para consultar la API de NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/api/).

Implementa la versión serial del autómata celular asegurando que la propagación del fuego respete las reglas de vecindad e ignición.

Implementa la versión paralela usando mpi4py, dividiendo el grid y manejando el intercambio de bordes en cada iteración temporal.

Añade medición de tiempos e incluye una forma básica de visualización (ej. snapshots en consola o matplotlib guardado en docs/assets/).

Fase 3: Implementación - Exercise 4 (Parallel K-Means - Covertype)Contexto: Dataset Covertype (https://archive.ics.uci.edu/ml/datasets/covertype).



Escribe el script para descargar y preprocesar el dataset Covertype.

Implementa K-Means en versión serial como baseline, fijando random seeds. Valida que asigne correctamente y actualice centroides.

Implementa el K-Means paralelo con mpi4py. Cada proceso debe calcular asignaciones locales y usar comunicación colectiva (MPI.COMM_WORLD.Allreduce) para sumar las coordenadas y contar los elementos de cada cluster antes de actualizar los centroides globales.

Añade logs de tiempo por iteración, tiempo total y validación de convergencia.

Fase 4: Scripts de Ejecución y Benchmarking



Crea comandos o scripts .sh para ejecutar ambos ejercicios variando el tamaño del problema y el número de procesos (ej. mpiexec -n 1, 2, 4, 8).

Genera un script en Python que lea los tiempos de ejecución, calcule el Speedup y la Efficiency, e imprima una tabla clara en consola para verificar si la paralelización es ventajosa.