#include <iostream>
#include <fstream>
#include <chrono>

const int N = 400; // Size of the square matrix
const int ITERATIONS = 10; // Number of iterations

// Function to apply Gaussian elimination on CPU
void gaussianElimination(float mat[N][N], float vec[N]) {
    for (int k = 0; k < N - 1; ++k) {
        for (int i = k + 1; i < N; ++i) {
            float factor = mat[i][k] / mat[k][k];
            for (int j = k; j < N; ++j) {
                mat[i][j] -= factor * mat[k][j];
            }
            vec[i] -= factor * vec[k];
        }
    }
}

// Function to solve the upper triangular system on CPU
void solveBackSubstitution(float mat[N][N], float vec[N], float sol[N]) {
    for (int i = N - 1; i >= 0; --i) {
        sol[i] = vec[i];
        for (int j = i + 1; j < N; ++j) {
            sol[i] -= mat[i][j] * sol[j];
        }
        sol[i] /= mat[i][i];
    }
}

int main() {
    std::ofstream outputFile("exec_times.txt");

    for (int i = 0; i < ITERATIONS; ++i) {
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

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Apply Gaussian elimination on CPU
        gaussianElimination(A, b);

        // Solve upper triangular system on CPU
        solveBackSubstitution(A, b, x);

        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;

        outputFile << "Ejecución " << i+1 << ": " << time << " segundos" << std::endl;
    }

    outputFile.close();

    return 0;
}
