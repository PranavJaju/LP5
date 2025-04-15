#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyKernel(int* matrixA, int* matrixB, int* resultMatrix, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; i++) {
            sum += matrixA[row * colsA + i] * matrixB[i * colsB + col];
        }
        resultMatrix[row * colsB + col] = sum;
    }
}

// Input matrix from user
void inputMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        cout << "Enter element " << i + 1 << ": ";
        cin >> matrix[i];
    }
}

// Display matrix
void printMatrix(int* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            cout << matrix[row * cols + col] << " ";
        }
        cout << '\n';
    }
    cout << '\n';
}

// Sequential matrix multiplication
void sequentialMatrixMultiply(int* matrixA, int* matrixB, int* resultMatrix, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            int sum = 0;
            for (int k = 0; k < colsA; k++) {
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            resultMatrix[i * colsB + j] = sum;
        }
    }
}

int main() {
    int rowsA, colsA, colsB;
    cout << "Enter the number of rows and columns of the first matrix: ";
    cin >> rowsA >> colsA;
    cout << "Enter the number of columns of the second matrix: ";
    cin >> colsB;

    int* matrixA = new int[rowsA * colsA];
    int* matrixB = new int[colsA * colsB];
    int* resultMatrix = new int[rowsA * colsB];

    inputMatrix(matrixA, rowsA, colsA);
    inputMatrix(matrixB, colsA, colsB);

    cout << "Matrix A:\n";
    printMatrix(matrixA, rowsA, colsA);

    cout << "Matrix B:\n";
    printMatrix(matrixB, colsA, colsB);

    int* devMatrixA;
    int* devMatrixB;
    int* devResultMatrix;

    cudaMalloc(&devMatrixA, rowsA * colsA * sizeof(int));
    cudaMalloc(&devMatrixB, colsA * colsB * sizeof(int));
    cudaMalloc(&devResultMatrix, rowsA * colsB * sizeof(int));

    cudaMemcpy(devMatrixA, matrixA, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devMatrixB, matrixB, colsA * colsB * sizeof(int), cudaMemcpyHostToDevice);

    int THREADS_PER_BLOCK = 16;
    int BLOCKS_X = (colsB + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int BLOCKS_Y = (rowsA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks(BLOCKS_X, BLOCKS_Y);

    // Sequential multiplication
    auto startTime = high_resolution_clock::now();
    sequentialMatrixMultiply(matrixA, matrixB, resultMatrix, rowsA, colsA, colsB);
    auto endTime = high_resolution_clock::now();
    auto sequentialDuration = duration_cast<microseconds>(endTime - startTime);

    cout << "Sequential Multiplication Result:\n";
    printMatrix(resultMatrix, rowsA, colsB);

    // Parallel multiplication
    startTime = high_resolution_clock::now();
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(devMatrixA, devMatrixB, devResultMatrix, rowsA, colsA, colsB);
    cudaMemcpy(resultMatrix, devResultMatrix, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);
    endTime = high_resolution_clock::now();
    auto parallelDuration = duration_cast<microseconds>(endTime - startTime);

    cout << "Parallel Multiplication Result:\n";
    printMatrix(resultMatrix, rowsA, colsB);

    cout << "Time taken for Sequential Multiplication: " << sequentialDuration.count() << " microseconds\n";
    cout << "Time taken for Parallel Multiplication: " << parallelDuration.count() << " microseconds\n";

    delete[] matrixA;
    delete[] matrixB;
    delete[] resultMatrix;

    cudaFree(devMatrixA);
    cudaFree(devMatrixB);
    cudaFree(devResultMatrix);

    return 0;
}
