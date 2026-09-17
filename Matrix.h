#pragma once

#include <cmath>
#include <concepts>
#include <utility>
#include <algorithm>
#include <random>
#include <cassert>
#include <numeric>
#include <vector>
#include <array>
#include <type_traits>

#include <blis/blis.h>

template <std::floating_point fp_type = float>
class Matrix
{
public:
	static Matrix zeros(std::size_t n_rows, std::size_t n_columns)
	{
		Matrix m{ n_rows, n_columns };
		std::fill(std::begin(m.buffer), std::end(m.buffer), static_cast<fp_type>(0));
		return m;
	}

	static Matrix random(std::size_t n_rows, std::size_t n_columns)
	{
		Matrix m{ n_rows, n_columns };
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<fp_type> nd;
		std::generate(std::begin(m.buffer), std::end(m.buffer), [&gen, &nd]() -> fp_type { return nd(gen); });
		return m;
	}

	template<std::size_t Tile_Size = 512>
	static void mul_B(Matrix& C, const Matrix& A, const Matrix& B, fp_type beta = static_cast<fp_type>(0))
	{
		assert(A.n_columns == B.n_rows && C.n_rows == A.n_rows && C.n_columns == B.n_columns);
		std::transform(
			std::begin(C.buffer), std::end(C.buffer), std::begin(C.buffer), 
			[&](const fp_type& v) { return beta * v; } 
		);
		for (std::size_t I = 0; I < C.n_rows; I += Tile_Size) {
			for (std::size_t J = 0; J < C.n_columns; J += Tile_Size) {
				for (std::size_t K = 0; K < A.n_columns; K += Tile_Size) {
					// Multiply the current tile
					for (std::size_t i = I; i < std::min(I + Tile_Size, C.n_rows); ++i) {
						for (std::size_t k = K; k < std::min(K + Tile_Size, A.n_columns); ++k) {
							std::transform(
								std::begin(C.buffer) + i * C.n_columns + J,
								std::begin(C.buffer) + i * C.n_columns + std::min(J + Tile_Size, C.n_columns),
								std::begin(B.buffer) + k * B.n_columns + J,
								std::begin(C.buffer) + i * C.n_columns + J,
								[u = A.buffer[A.n_columns * i + k]](const fp_type& v1, const fp_type& v2) { return v1 + v2 * u; }
							);
							//for (std::size_t j = J; j < std::min(J + Tile_Size, C.n_columns); ++j) {
							//	C(i, j) += A(i, k) * B(k, j);
							//}
						}
					}
				}
			}
		}
	}

	static void mul_BLIS(Matrix& C, const Matrix& A, const Matrix& B, fp_type beta = static_cast<fp_type>(0))
	{
		static_assert(std::is_same_v<fp_type, float>);

		const float alpha = 1.0f;

		bli_sgemm(
			BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, A.rows(), B.columns(), A.columns(), 
			&alpha, 
			A.buffer.data(), A.columns(), 1,
			B.buffer.data(), B.columns(), 1,
			&beta,
			C.buffer.data(), C.columns(), 1
		);
	}

	const fp_type& operator() (std::size_t r, std::size_t c) const
	{
		return buffer[n_columns * r + c];
	}

	fp_type& operator() (std::size_t r, std::size_t c)
	{
		return buffer[n_columns * r + c];
	}

	size_t rows() const { return n_rows; }
	size_t columns() const { return n_columns; }
private:
	Matrix(std::size_t n_rows, std::size_t n_columns) : n_rows(n_rows), n_columns(n_columns), buffer(n_rows* n_columns) {}
	const size_t n_rows, n_columns;
	std::vector<fp_type> buffer;
};

namespace MatrixDetail
{
	// Processor details
	namespace Processor
	{
		constexpr size_t Alignment = 32;
		constexpr size_t N_Registers = 16;
		constexpr size_t Width = 32;
	}

	template <std::floating_point fp_type>
	struct Sizes
	{
		static constexpr size_t MR = 6;
		static constexpr size_t NR = 2 * Processor::Width / sizeof(fp_type);

		// These should probably be determined by L2 cache size;
		// prioritize increasing KC over MC.
		static constexpr size_t MC = 96;
		static constexpr size_t KC = 1024;

		static constexpr size_t NC = 4096;

		static constexpr size_t Packed_A_Size = MC * KC;
		static constexpr size_t Packed_B_Size = KC * NC;

		static_assert(MC % MR == 0 && NC % NR == 0);
	};

	template<std::floating_point fp_type>
	void pack_B(std::array<fp_type, Sizes<fp_type>::Packed_B_Size>& packed,
				const Matrix<fp_type>& B, size_t r_start, size_t c_start)
	{
		using Sz = Sizes<fp_type>;

		for (size_t i = 0; i < Sz::Packed_B_Size; ++i) {
			const size_t B_row = r_start + (i / Sz::NR) % Sz::KC;
			const size_t B_column = c_start + Sz::NR * (i / (Sz::NR * Sz::KC)) + (i % Sz::NR);
			packed[i] = (B_row < B.rows() && B_column < B.columns()) ?
				B(B_row, B_column) : static_cast<fp_type>(0);
		}
	}

	template<std::floating_point fp_type>
	void pack_A(std::array<fp_type, Sizes<fp_type>::Packed_A_Size>& packed,
				const Matrix<fp_type>& A, size_t r_start, size_t c_start)
	{
		using Sz = Sizes<fp_type>;

		for (size_t i = 0; i < Sz::Packed_A_Size; ++i) {
			const size_t A_row = r_start + Sz::MR * (i / (Sz::KC * Sz::MR)) + (i % Sz::MR);
			const size_t A_column = c_start + (i / Sz::MR) % Sz::KC;
			packed[i] = (A_row < A.rows() && A_column < A.columns()) ?
				A(A_row, A_column) : static_cast<fp_type>(0);
		}
	}

	template<std::floating_point fp_type>
	void micro_kernel_generic(Matrix<fp_type>& C, size_t r_start, size_t c_start,
					  const std::array<fp_type, Sizes<fp_type>::Packed_A_Size>& A,
					  const std::array<fp_type, Sizes<fp_type>::Packed_B_Size>& B,
					  size_t A_start, size_t B_start, fp_type beta, size_t K_lim)
	{
		using Sz = Sizes<fp_type>;

		// row-major stored auxiliary panel for C
		alignas(Processor::Alignment) std::array<fp_type, Sz::MR * Sz::NR> aux = {};

		// multiply A and B panels into aux
		for (size_t i = 0; i < K_lim; ++i) {
			for (size_t j = 0; j < Sz::MR; ++j) {
				for (size_t k = 0; k < Sz::NR; ++k) {
					aux[j * Sz::NR + k] += A[A_start + j] * B[B_start + k];
				}
			}
			A_start += Sz::MR;
			B_start += Sz::NR;
		}

		// update C
		for (size_t i = 0; i < Sz::MR; ++i) {
			for (size_t j = 0; j < Sz::NR; ++j) {
				if (r_start + i < C.rows() && c_start + j < C.columns()) {
					C(r_start + i, c_start + j) = std::fma(beta, C(r_start + i, c_start + j), aux[i * Sz::NR + j]);
				}
			}
		}
	}

	void micro_kernel_f32_avx2(Matrix<float>& C, size_t r_start, size_t c_start,
							   const std::array<float, Sizes<float>::Packed_A_Size>& packed_A,
							   const std::array<float, Sizes<float>::Packed_B_Size>& packed_B,
							   size_t A_start, size_t B_start, float beta, size_t K_lim);

	template<std::floating_point fp_type>
	void macro_kernel(Matrix<fp_type>& C, size_t r_start, size_t c_start,
					  const std::array<fp_type, Sizes<fp_type>::Packed_A_Size>& A,
					  const std::array<fp_type, Sizes<fp_type>::Packed_B_Size>& B,
					  fp_type beta, size_t K_lim)
	{
		using Sz = Sizes<fp_type>;

		for (size_t i = 0; i < std::min(Sz::NC, C.columns() - c_start); i += Sz::NR) { // JR loop
			for (size_t j = 0; j < std::min(Sz::MC, C.rows() - r_start); j += Sz::MR) { // IR loop
				if constexpr (std::is_same_v<fp_type, float>)
					micro_kernel_f32_avx2(C, r_start + j, c_start + i, A, B, j * Sz::KC, i * Sz::KC, beta, K_lim);
				else
					micro_kernel_generic(C, r_start + j, c_start + i, A, B, j * Sz::KC, i * Sz::KC, beta, K_lim);
			}
		}
	}
}

template<std::floating_point fp_type>
void mul_C(Matrix<fp_type>& C, const Matrix<fp_type>& A, const Matrix<fp_type>& B, fp_type beta)
{
	using namespace MatrixDetail;
	using Sz = Sizes<fp_type>;

	alignas(Processor::Alignment) thread_local std::array<fp_type, Sz::Packed_A_Size> packed_A;
	alignas(Processor::Alignment) thread_local std::array<fp_type, Sz::Packed_B_Size> packed_B;

	assert(A.columns() == B.rows() && C.rows() == A.rows() && C.columns() == B.columns());
	const size_t M = A.rows(), N = B.columns(), K = A.columns();

	for (size_t i = 0; i < N; i += Sz::NC) { // JC loop
		for (size_t j = 0; j < K; j += Sz::KC) { // PC loop
			pack_B(packed_B, B, j, i);
			for (size_t k = 0; k < M; k += Sz::MC) { // IC loop
				pack_A(packed_A, A, k, j);
				// Only apply beta on the first iteration over this slice of C,
				// before intermediate values have been added.
				macro_kernel(C, k, i, packed_A, packed_B, (j == 0) ? beta : static_cast<fp_type>(1), std::min(Sz::KC, K));
				// For small matrices, K may be much smaller than KC, so use the smaller of the two
				// in the micro-kernel loop to prevent unnecessary computations.
			}
		}
	}
}
