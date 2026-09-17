#include "Matrix.h"

#include <chrono>
#include <iostream>
#include <format>

#include <blis/blis.h>

int main(int argc, char* argv[])
{
	if (argc < 4)
		return EXIT_FAILURE;
	
	Matrix A = Matrix<>::random(std::stoul(argv[1]), std::stoul(argv[2]));
	Matrix B = Matrix<>::random(std::stoul(argv[2]), std::stoul(argv[3]));

	Matrix resultA = Matrix<>::random(A.rows(), B.columns());
	const auto startA = std::chrono::steady_clock::now();
	mul_C<float>(resultA, A, B, 0.0f);
	const auto endA = std::chrono::steady_clock::now();
	std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(endA - startA) << "\n";
	double f = 2.0 * A.rows() * A.columns() * B.columns();
	double d = std::chrono::duration<double>(endA - startA).count();
	std::cout << (f / d) * 1e-9 << " GFLOPS\n";

	Matrix resultT = Matrix<>::random(A.rows(), B.columns());
	const auto startT = std::chrono::steady_clock::now();
	Matrix<>::mul_BLIS(resultT, A, B);
	const auto endT = std::chrono::steady_clock::now();
	std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(endT - startT) << "\n";
	d = std::chrono::duration<double>(endT - startT).count();
	std::cout << (f / d) * 1e-9 << " GFLOPS\n";

	Matrix resultB = Matrix<>::random(A.rows(), B.columns());
	const auto startB = std::chrono::steady_clock::now();
	Matrix<>::mul_B(resultB, A, B);
	const auto endB = std::chrono::steady_clock::now();
	std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(endB - startB) << "\n";
	d = std::chrono::duration<double>(endB - startB).count();
	std::cout << (f / d) * 1e-9 << " GFLOPS\n";

	const float EPSILON = 0.01f;
	for (std::size_t i = 0; i < resultT.rows(); ++i) {
		for (std::size_t j = 0; j < resultT.columns(); ++j) {
			if (resultT(i, j) - resultA(i, j) > EPSILON || resultT(i, j) - resultB(i, j) > EPSILON) {
				std::cout << std::format("Discrepancy found! i = {} j = {} T = {} A = {} B = {}\n", i, j, resultT(i, j), resultA(i, j), resultB(i, j));
			}
		}
	}

	return 0;
}
