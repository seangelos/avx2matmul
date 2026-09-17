This is a simple implementation of Goto-style matrix multiplication, using a layered approach in the fashion of the BLIS framework.

In particular, the "block-panel" algorithm is implemented as a set of portable functions for iterating over matrices and packing into specially crafted auxiliary buffers. Architectural details are confined only to the innermost level of computation, dubbed the "micro-kernel", here written using AVX2 intrinsics.

This code was written as part of a microarchitectures course. As such, it is not a general GEMM operation, only supporting row-major storage order and α=1. It has only been tested and optimized on Intel "Haswell" microarchitecture and achieves peak performance for large and square-ish matrices. For smaller matrices, packing overhead becomes substantial and more specialized algorithms should be used.

## Performance

Testing was performed on Fedora 44, using the Clang 22.1.8 compiler, BLIS 2.0 and an Intel Core i7-4790 CPU (theoretical max. performance of 115.2 GFLOPS). All results are expressed in GFLOPS, and since there was quite a bit of run-to-run variance the best of 5 runs has been taken.

|matrix size|"naive" C++ baseline|C++ micro-kernel|AVX2 micro-kernel|BLIS |
|-----------|--------------------|----------------|-----------------|-----|
|512²       |16.74               |27.27           |31.84            |72.56|
|1024²      |15.07               |60.54           |69.46            |89.37|
|2048²      |15.31               |75.93           |83.95            |95.79|
|4096²      |10.43               |79.11           |90.80            |92.85|

By hand-writing the micro-kernel using AVX2 intrinsics, an increase in the order of 10-15% versus a generic C++ micro-kernel has been achieved. A takeaway from this is that the the vectorizer is good enough (on Clang at least) to produce a highly optimized micro-kernel from C++ code, only 15-20% slower than BLIS for sufficiently large sizes.

## Citations and links

1. K. Goto and R. A. Van De Geijn (2008) [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~pingali/CSE392/2011sp/lectures/a12-goto.pdf)
2. F. G. Van Zee and R. A. Van De Geijn (2015) [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf)
3. [Useful illustration of MMBP algorithm from BLIS documentation](https://github.com/flame/blis/blob/master/docs/diagrams/mmbp_algorithm_color.pdf)
