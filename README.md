# Taller 1 - CUDA

## Descripción
Este material corresponde al Taller 1 en el curso CE 4302 — Arquitectura de Computadores II, diseñado con el objetivo de comprender el uso de CUDA en programación paralela. El taller incluye ejercicios para realizar operaciones de suma de vectores, multiplicación de matrices y eliminación gaussiana utilizando tanto CUDA como implementaciones seriales en C++.

# Prerrequisitos
- CUDA Toolkit instalado en el sistema.
- Configuración de CUDA en Visual Studio para ejecutar el código CUDA.
- Presencia de MINGW64 para ejecutar el código en C++.

## Uso
El código se encuentra organizado en la carpeta src/, la cual contiene dos archivos y una carpeta:

### Archivos:

#### vecadd.cu
Realiza la operación de suma elemento por elemento entre los vectores a y b, almacenando el resultado en el vector c. Además, compara el rendimiento en al menos 5 casos diferentes. Para ejecutarlo, sigue los pasos:

```bash
nvcc vecadd.cu -o vecadd
./vecadd
```
#### matrixMultiplication.cu

Calcula el resultado de la multiplicación de dos matrices de tamaño 4x4, utilizando paralelismo con CUDA. Para ejecutarlo:

```bash
nvcc matrixMultiplication.cu -o matrixMultiplication
./matrixMultiplication
```

### Carpeta "gaussian-elimination":

En esta carpeta se encuentran dos archivos:

#### gaussianElimination.cpp
Realiza la eliminación gaussiana de manera serial sin utilizar paralelismo. Guarda los resultados en el archivo exec_time.txt. Para ejecutarlo:

```bash
g++ gaussianElimination.cpp -o gaussianElimination
./gaussianElimination
```

#### gaussianElimination.cu
Realiza la eliminación gaussiana utilizando CUDA. Guarda los resultados en el archivo exec_times_cuda.txt. Para ejecutarlo:

```bash
nvcc gaussianElimination.cu -o gaussianEliminationCUDA
./gaussianEliminationCUDA
```

## Detalles de implementación
Las pruebas fueron desarrolladas y evaluadas en un sistema con GPU NVIDIA GeForce RTX 3060 Ti y CPU 12th Gen Intel(R) Core(TM) i5-12400.

# Autores
- Carrillo Salazar Juan Pablo
- Esquivel Sanchez Jonathan Daniel,
- Mendoza Mata Jose Fabian,
- Ortiz Vega Angelo Jesus

# Fecha
31/03/2024

# Repositorio
Encuentra el proyecto completo en: [GitHub](https://github.com/ce-itcr/cuda-performance-testing).
