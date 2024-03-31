#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <ctime>
#include <iostream>

const int N = 100; // Size of the square matrix

// Kernel to apply Gaussian elimination in parallel
__global__ void gaussianEliminationKernel(float* mat, float* vec, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > k && i < N) {
        float factor = mat[i * N + k] / mat[k * N + k];
        for (int j = k; j < N; ++j) {
            mat[i * N + j] -= factor * mat[k * N + j];
        }
        vec[i] -= factor * vec[k];
    }
}

// Function to apply Gaussian elimination on GPU
void gaussianEliminationCUDA(float* d_mat, float* d_vec) {
    for (int k = 0; k < N - 1; ++k) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        gaussianEliminationKernel <<< blocksPerGrid, threadsPerBlock >>> (d_mat, d_vec, k);
        cudaDeviceSynchronize();
    }
}

// Function to solve the upper triangular system on CPU
void solveBackSubstitution(float* mat, float* vec, float* sol) {
    for (int i = N - 1; i >= 0; --i) {
        sol[i] = vec[i];
        for (int j = i + 1; j < N; ++j) {
            sol[i] -= mat[i * N + j] * sol[j];
        }
        sol[i] /= mat[i * N + i];
    }
}

int main() {
    float A[N][N]; // Coefficient matrix on host
    float b[N];    // Independent terms vector on host
    float x[N];    // Solution vector on host

    // Initialization of the matrix and vector
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = (i == j) ? 1.0 : 0.0; // Identity matrix
        }
        b[i] = i + 1; // Independent terms vector
    }

    // Memory allocation on the device
    float* d_mat, * d_vec;
    cudaMalloc((void**)&d_mat, N * N * sizeof(float));
    cudaMalloc((void**)&d_vec, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_mat, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Measure execution time
    std::clock_t start = std::clock();

    // Apply Gaussian elimination on GPU
    gaussianEliminationCUDA(d_mat, d_vec);

    // Copy results back to host
    cudaMemcpy(A, d_mat, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_vec, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Solve upper triangular system on CPU
    solveBackSubstitution((float*)A, b, x);

    std::clock_t end = std::clock();
    double time = (end - start) / (double)CLOCKS_PER_SEC;

    std::cout << "Execution time (CUDA): " << time << " seconds" << std::endl;

    // Free memory on the device
    cudaFree(d_mat);
    cudaFree(d_vec);

    return 0;
}
