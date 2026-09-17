#include "Matrix.h"

#include <array>
#include <climits>
#include <immintrin.h>

namespace MatrixDetail
{
	void micro_kernel_f32_avx2(Matrix<float>& C, size_t r_start, size_t c_start,
							   const std::array<float, Sizes<float>::Packed_A_Size>& packed_A,
							   const std::array<float, Sizes<float>::Packed_B_Size>& packed_B,
							   size_t A_start, size_t B_start, float beta, size_t K_lim)
	{
		using Sz = Sizes<float>;
		static_assert(Sz::NR == 16 && Sz::MR == 6);

		// This seems to help by a tiny amount.
		// _MM_HINT_ET0 is supposed to generate PREFETCHW which might be more
		// appropriate, but MSVC doesn't define it and GCC seems to ignore it.
		_mm_prefetch(reinterpret_cast<const char *>(&C(r_start, c_start)), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char *>(&C(r_start + 1, c_start)), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char *>(&C(r_start + 2, c_start)), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char *>(&C(r_start + 3, c_start)), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char *>(&C(r_start + 4, c_start)), _MM_HINT_T0);
		_mm_prefetch(reinterpret_cast<const char *>(&C(r_start + 5, c_start)), _MM_HINT_T0);

		// GCC seems to emit weird forms of vfma in the loop below, which causes
		// a performance regression. Register spilling maybe?
		std::array<__m256, 2 * Sz::MR> aux = {};

		const size_t K_iter = K_lim / 4; // iterations of unrolled loop
		const size_t K_rem  = K_lim % 4; // remaining iterations

		for (size_t i = 0; i < K_iter; ++i) {
			// inner loop iteration 0
			_mm_prefetch(reinterpret_cast<const char *>(packed_A.data() + A_start + 64), _MM_HINT_T0);
			__m256 B_vec0 = _mm256_load_ps(packed_B.data() + B_start);
			__m256 B_vec1 = _mm256_load_ps(packed_B.data() + B_start + 8);

			__m256 A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start);
			aux[0] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[0]);
			aux[1] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[1]);
			__m256 A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 1);
			aux[2] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[2]);
			aux[3] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[3]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 2);
			aux[4] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[4]);
			aux[5] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[5]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 3);
			aux[6] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[6]);
			aux[7] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[7]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 4);
			aux[8] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[8]);
			aux[9] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[9]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 5);
			aux[10] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[10]);
			aux[11] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[11]);

			// iteration 1
			B_vec0 = _mm256_load_ps(packed_B.data() + B_start + 16);
			B_vec1 = _mm256_load_ps(packed_B.data() + B_start + 24);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 6);
			aux[0] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[0]);
			aux[1] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[1]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 7);
			aux[2] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[2]);
			aux[3] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[3]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 8);
			aux[4] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[4]);
			aux[5] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[5]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 9);
			aux[6] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[6]);
			aux[7] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[7]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 10);
			aux[8] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[8]);
			aux[9] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[9]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 11);
			aux[10] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[10]);
			aux[11] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[11]);

			// iteration 2
			_mm_prefetch(reinterpret_cast<const char *>(packed_A.data() + A_start + 76), _MM_HINT_T0);
			B_vec0 = _mm256_load_ps(packed_B.data() + B_start + 32);
			B_vec1 = _mm256_load_ps(packed_B.data() + B_start + 40);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 12);
			aux[0] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[0]);
			aux[1] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[1]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 13);
			aux[2] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[2]);
			aux[3] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[3]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 14);
			aux[4] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[4]);
			aux[5] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[5]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 15);
			aux[6] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[6]);
			aux[7] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[7]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 16);
			aux[8] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[8]);
			aux[9] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[9]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 17);
			aux[10] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[10]);
			aux[11] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[11]);

			// iteration 3
			B_vec0 = _mm256_load_ps(packed_B.data() + B_start + 48);
			B_vec1 = _mm256_load_ps(packed_B.data() + B_start + 56);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 18);
			aux[0] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[0]);
			aux[1] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[1]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 19);
			aux[2] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[2]);
			aux[3] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[3]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 20);
			aux[4] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[4]);
			aux[5] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[5]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 21);
			aux[6] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[6]);
			aux[7] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[7]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 22);
			aux[8] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[8]);
			aux[9] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[9]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 23);
			aux[10] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[10]);
			aux[11] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[11]);

			A_start += 4 * Sz::MR;
			B_start += 4 * Sz::NR;
		}

		for (size_t i = 0; i < K_rem; ++i) {
			__m256 B_vec0 = _mm256_load_ps(packed_B.data() + B_start);
			__m256 B_vec1 = _mm256_load_ps(packed_B.data() + B_start + 8);

			__m256 A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start);
			aux[0] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[0]);
			aux[1] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[1]);
			__m256 A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 1);
			aux[2] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[2]);
			aux[3] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[3]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 2);
			aux[4] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[4]);
			aux[5] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[5]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 3);
			aux[6] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[6]);
			aux[7] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[7]);

			A_col_bc0 = _mm256_broadcast_ss(packed_A.data() + A_start + 4);
			aux[8] = _mm256_fmadd_ps(A_col_bc0, B_vec0, aux[8]);
			aux[9] = _mm256_fmadd_ps(A_col_bc0, B_vec1, aux[9]);
			A_col_bc1 = _mm256_broadcast_ss(packed_A.data() + A_start + 5);
			aux[10] = _mm256_fmadd_ps(A_col_bc1, B_vec0, aux[10]);
			aux[11] = _mm256_fmadd_ps(A_col_bc1, B_vec1, aux[11]);

			A_start += Sz::MR;
			B_start += Sz::NR;
		}

		// update C
		if (beta != 0.0f) {
			const __m256 beta_v = _mm256_set1_ps(beta);
			if (c_start + 15 < C.columns()) {
				for (size_t i = 0; i < Sz::MR; ++i) {
					if (r_start + i < C.rows()) {
						__m256 v0 = _mm256_loadu_ps(&C(r_start + i, c_start));
						__m256 v1 = _mm256_loadu_ps(&C(r_start + i, c_start + 8));
						v0 = _mm256_fmadd_ps(beta_v, v0, aux[2 * i]);
						v1 = _mm256_fmadd_ps(beta_v, v1, aux[2 * i + 1]);
						_mm256_storeu_ps(&C(r_start + i, c_start), v0);
						_mm256_storeu_ps(&C(r_start + i, c_start + 8), v1);
					}
				}
			} else if (c_start < C.columns()) {
				// Determine which columns need to be discarded, and set mask bit accordingly.
				std::array<int, 8> mv = {};
				for (size_t i = 0; i < (C.columns() - c_start) % 8; ++i)
					mv[i] = INT_MIN;
				__m256i mask = _mm256_set_epi32(mv[7], mv[6], mv[5], mv[4], mv[3], mv[2], mv[1], mv[0]);
				if (c_start + 7 < C.columns()) { // 8 columns fit cleanly; use mask for other 8
					for (size_t i = 0; i < Sz::MR; ++i) {
						if (r_start + i < C.rows()) {
							// If the number of columns is divisible by 8, an extra
							// mask store is performed with a mask of all zeros.
							// I'm not sure if this is worth addressing.
							__m256 v0 = _mm256_loadu_ps(&C(r_start + i, c_start));
							__m256 v1 = _mm256_maskload_ps(&C(r_start + i, c_start + 8), mask);
							v0 = _mm256_fmadd_ps(beta_v, v0, aux[2 * i]);
							v1 = _mm256_fmadd_ps(beta_v, v1, aux[2 * i + 1]);
							_mm256_storeu_ps(&C(r_start + i, c_start), v0);
							_mm256_maskstore_ps(&C(r_start + i, c_start + 8), mask, v1);
						}
					}
				} else { // less than 8 columns fit
					for (size_t i = 0; i < Sz::MR; ++i) {
						if (r_start + i < C.rows()) {
							__m256 v = _mm256_maskload_ps(&C(r_start + i, c_start), mask);
							v = _mm256_fmadd_ps(beta_v, v, aux[2 * i]);
							_mm256_maskstore_ps(&C(r_start + i, c_start), mask, v);
						}
					}
				}
			}
		} else { // don't load C if beta is 0 (first iteration of PC loop only)
			if (c_start + 15 < C.columns()) {
				for (size_t i = 0; i < Sz::MR; ++i) {
					if (r_start + i < C.rows()) {
						_mm256_storeu_ps(&C(r_start + i, c_start), aux[2 * i]);
						_mm256_storeu_ps(&C(r_start + i, c_start + 8), aux[2 * i + 1]);
					}
				}
			} else if (c_start < C.columns()) {
				std::array<int, 8> mv = {};
				for (size_t i = 0; i < (C.columns() - c_start) % 8; ++i)
					mv[i] = INT_MIN;
				__m256i mask = _mm256_set_epi32(mv[7], mv[6], mv[5], mv[4], mv[3], mv[2], mv[1], mv[0]);
				if (c_start + 7 < C.columns()) { // 8 columns fit cleanly; use mask for other 8
					for (size_t i = 0; i < Sz::MR; ++i) {
						if (r_start + i < C.rows()) {
							_mm256_storeu_ps(&C(r_start + i, c_start), aux[2 * i]);
							_mm256_maskstore_ps(&C(r_start + i, c_start + 8), mask, aux[2 * i + 1]);
						}
					}
				} else { // less than 8 columns fit
					for (size_t i = 0; i < Sz::MR; ++i) {
						if (r_start + i < C.rows()) {
							_mm256_maskstore_ps(&C(r_start + i, c_start), mask, aux[2 * i]);
						}
					}
				}
			}
		}
	}
}
