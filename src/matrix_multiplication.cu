#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <ctime>

#define N 4 // Size of the matrices

// Kernel for matrix multiplication
__global__ void matrixMultiply(int* a, int* b, int* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < N; ++k) {
        sum += a[row * N + k] * b[k * N + col];
    }

    c[row * N + col] = sum;
}

int main() {
    int a[N][N], b[N][N], c[N][N]; // Host matrices
    int* dev_a, * dev_b, * dev_c;    // Device matrices

    // Initialization of matrices a and b
    printf("Matrix A:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = i + j;
            printf("%d\t", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Matrix B:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i][j] = i - j; 
            printf("%d\t", b[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);

    // Call the kernel
    matrixMultiply <<< blocksPerGrid, threadsPerBlock >>> (dev_a, dev_b, dev_c);

    // Copy the result from the device to the host
    cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Show the result
    printf("Result of matrix multiplication:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }

    // Free memory on the device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
