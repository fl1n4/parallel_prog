#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <Eigen/Dense>
#include <mpi.h>  // [MPI]

using namespace std;
using namespace Eigen;

// Генерация случайной целочисленной матрицы (плоский вектор)
vector<int> generateRandomIntMatrix(int rows, int cols, int min_val = 0, int max_val = 10) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(min_val, max_val);

    vector<int> matrix(rows * cols);
    for (int i = 0; i < rows * cols; ++i)
        matrix[i] = dist(gen);

    return matrix;
}

// Запись плоской матрицы в файл
void writeMatrixToFile(const string& filename, const vector<int>& matrix, int rows, int cols) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    file << rows << " " << cols << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << setw(6) << matrix[i * cols + j] << " ";
        }
        file << endl;
    }
}

// Верификация с помощью Eigen (только на rank 0)
void verifyWithEigen(const vector<int>& A, const vector<int>& B, const vector<int>& C, int N) {
    MatrixXi eigenA(N, N);
    MatrixXi eigenB(N, N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            eigenA(i, j) = A[i * N + j];
            eigenB(i, j) = B[i * N + j];
        }

    MatrixXi eigenC = eigenA * eigenB;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (C[i * N + j] != eigenC(i, j)) {
                throw runtime_error("Verification failed at [" + to_string(i) + "][" + to_string(j) + "]");
            }
        }
}

// Запись результата
void writeFullResults(const string& filename, const vector<int>& result, double time_ms, int N) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    file << "Matrix size: " << N << "x" << N << endl;
    file << "Execution time: " << time_ms << " ms" << endl;
    file << "Result matrix (" << N << "x" << N << "):\n";

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file << setw(8) << result[i * N + j] << " ";
        }
        file << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int min_size = 100;
    const int max_size = 800;
    const int step = 10;

    try {
        for (int matrix_size = min_size; matrix_size <= max_size; matrix_size += step) {
            int rows_per_proc = matrix_size / world_size;
            int remainder = matrix_size % world_size;

            int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);

            vector<int> A_full, B, C_full;
            vector<int> A_local(my_rows * matrix_size);
            vector<int> C_local(my_rows * matrix_size, 0);

            if (rank == 0) {
                A_full = generateRandomIntMatrix(matrix_size, matrix_size, 0, 10);
                B = generateRandomIntMatrix(matrix_size, matrix_size, 0, 10);

                writeMatrixToFile("mpi_matrix_A_" + to_string(matrix_size) + ".txt", A_full, matrix_size, matrix_size);
                writeMatrixToFile("mpi_matrix_B_" + to_string(matrix_size) + ".txt", B, matrix_size, matrix_size);
            }
            else {
                B.resize(matrix_size * matrix_size);
            }

            // Рассылаем матрицу B
            MPI_Bcast(B.data(), matrix_size * matrix_size, MPI_INT, 0, MPI_COMM_WORLD);

            // Подготовка sendcounts и displs для Scatterv
            vector<int> sendcounts(world_size), displs(world_size);
            int offset = 0;
            for (int i = 0; i < world_size; ++i) {
                int rows = rows_per_proc + (i < remainder ? 1 : 0);
                sendcounts[i] = rows * matrix_size;
                displs[i] = offset;
                offset += sendcounts[i];
            }

            // Рассылаем части A
            MPI_Scatterv(
                rank == 0 ? A_full.data() : nullptr,
                sendcounts.data(),
                displs.data(),
                MPI_INT,
                A_local.data(),
                my_rows * matrix_size,
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            // Локальное перемножение
            auto start = chrono::high_resolution_clock::now();

            for (int i = 0; i < my_rows; ++i) {
                for (int j = 0; j < matrix_size; ++j) {
                    int sum = 0;
                    for (int k = 0; k < matrix_size; ++k) {
                        sum += A_local[i * matrix_size + k] * B[k * matrix_size + j];
                    }
                    C_local[i * matrix_size + j] = sum;
                }
            }

            auto end = chrono::high_resolution_clock::now();
            double local_time = chrono::duration<double, milli>(end - start).count();

            // Собираем результат
            if (rank == 0)
                C_full.resize(matrix_size * matrix_size);

            MPI_Gatherv(
                C_local.data(),
                my_rows * matrix_size,
                MPI_INT,
                rank == 0 ? C_full.data() : nullptr,
                sendcounts.data(),
                displs.data(),
                MPI_INT,
                0,
                MPI_COMM_WORLD
            );

            // Верификация и запись результата
            if (rank == 0) {
                double max_time;
                MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                verifyWithEigen(A_full, B, C_full, matrix_size);

                writeFullResults("mpi_result_" + to_string(matrix_size) + ".txt", C_full, max_time, matrix_size);

                cout << "Done size " << matrix_size << "x" << matrix_size << ". Time: " << max_time << " ms" << endl;
            }
            else {
                MPI_Reduce(&local_time, nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            }
        }

        if (rank == 0)
            cout << "All tests completed!" << endl;
    }
    catch (const exception& e) {
        cerr << "Rank " << rank << " error: " << e.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
