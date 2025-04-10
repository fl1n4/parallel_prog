#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Генерация случайной целочисленной матрицы
vector<vector<int>> generateRandomIntMatrix(int rows, int cols, int min_val = 0, int max_val = 10) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(min_val, max_val);

    vector<vector<int>> matrix(rows, vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(gen);
        }
    }
    return matrix;
}

// Запись матрицы в файл
void writeMatrixToFile(const string& filename, const vector<vector<int>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    file << matrix.size() << " " << (matrix.empty() ? 0 : matrix[0].size()) << endl;
    for (const auto& row : matrix) {
        for (int val : row) {
            file << setw(6) << val << " ";
        }
        file << endl;
    }
}

// Перемножение целочисленных матриц
vector<vector<int>> multiplyIntMatrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    vector<vector<int>> C(m, vector<int>(p, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Верификация с помощью Eigen
void verifyWithEigen(const vector<vector<int>>& A, const vector<vector<int>>& B, const vector<vector<int>>& C) {
    MatrixXi eigenA(A.size(), A[0].size());
    MatrixXi eigenB(B.size(), B[0].size());

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); ++j) {
            eigenA(i, j) = A[i][j];
        }
    }

    for (int i = 0; i < B.size(); ++i) {
        for (int j = 0; j < B[0].size(); ++j) {
            eigenB(i, j) = B[i][j];
        }
    }

    MatrixXi eigenC = eigenA * eigenB;

    for (int i = 0; i < C.size(); ++i) {
        for (int j = 0; j < C[0].size(); ++j) {
            if (C[i][j] != eigenC(i, j)) {
                throw runtime_error("Verification failed at [" + to_string(i) + "][" + to_string(j) + "]");
            }
        }
    }
}

// Запись полных результатов
void writeFullResults(const string& filename,
    const vector<vector<int>>& result,
    double time_ms,
    int size) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    file << "Matrix size: " << size << "x" << size << endl;
    file << "Execution time: " << time_ms << " ms" << endl;
    file << "Result matrix (" << result.size() << "x" << result[0].size() << "):\n";

    for (const auto& row : result) {
        for (int val : row) {
            file << setw(8) << val << " ";
        }
        file << endl;
    }
}

int main() {
    const int min_size = 100;
    const int max_size = 1000;
    const int step = 10;

    try {
        for (int size = min_size; size <= max_size; size += step) {
            cout << "Processing size " << size << "x" << size << "... ";

            // Генерация матриц
            auto A = generateRandomIntMatrix(size, size, 0, 10);
            auto B = generateRandomIntMatrix(size, size, 0, 10);

            // Сохранение исходных данных
            writeMatrixToFile("int_matrix_A_" + to_string(size) + ".txt", A);
            writeMatrixToFile("int_matrix_B_" + to_string(size) + ".txt", B);

            // Измерение времени
            auto start = chrono::high_resolution_clock::now();
            auto C = multiplyIntMatrices(A, B);
            auto end = chrono::high_resolution_clock::now();
            double time_ms = chrono::duration<double, milli>(end - start).count();

            // Проверка
            verifyWithEigen(A, B, C);

            // Сохранение результатов
            writeFullResults("int_result_" + to_string(size) + ".txt", C, time_ms, size);

            cout << "Done. Time: " << time_ms << " ms" << endl;
        }
        cout << "All tests completed successfully!" << endl;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}