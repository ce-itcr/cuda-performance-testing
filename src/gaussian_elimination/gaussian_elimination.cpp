#include <iostream>
#include <ctime>

const int MaxN = 1000; // Maximum size of the square matrix

// Function to print a matrix
void printMatrix(float mat[MaxN][MaxN], int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << mat[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Gaussian elimination on a matrix
void gaussianElimination(float mat[MaxN][MaxN], float vec[MaxN], int N) {
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

// Solve the upper triangular system
void solveBackSubstitution(float mat[MaxN][MaxN], float vec[MaxN], float sol[MaxN], int N) {
    for (int i = N - 1; i >= 0; --i) {
        sol[i] = vec[i];
        for (int j = i + 1; j < N; ++j) {
            sol[i] -= mat[i][j] * sol[j];
        }
        sol[i] /= mat[i][i];
    }
}

int main() {
    float A[MaxN][MaxN]; // Coefficient matrix
    float b[MaxN];    // Independent terms vector
    float x[MaxN];    // Solution vector

    for (int N = 100; N <= MaxN; N += 100) { // Testing with different sizes of matrices
        // Initialization of the matrix and vector
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = (i == j) ? 1.0 : 0.0; // Identity matrix
            }
            b[i] = i + 1; // Independent terms vector
        }

        // Measure execution time
        std::clock_t start = std::clock();

        // Apply Gaussian elimination
        gaussianElimination(A, b, N);

        // Solve the upper triangular system
        solveBackSubstitution(A, b, x, N);

        std::clock_t end = std::clock();
        double time = (end - start) / (double)CLOCKS_PER_SEC;

        std::cout << "Execution time for N = " << N << ": " << time << " seconds" << std::endl;
    }

    return 0;
}
