# ulmBLAS

**A high performance BLAS implementation**

---

This library is part of my lecture:

**Software Basics for High Performance Computing** (MATH9367)  
Ulm University

- [Course page for ulmBLAS](http://www.mathematik.uni-ulm.de/~lehn/ulmBLAS)
- [General course page](http://www.mathematik.uni-ulm.de/~lehn/sghpc)

---

And yes, I am particularly proud of the section demonstrating how to achieve
peak performance for the matrix-matrix product:

👉 [Achieving peak performance for GEMM](http://www.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html)

---

**Note:** Further development will take place in
[ulmBLAS-core](https://github.com/michael-lehn/ulmBLAS-core).

---
## Background

The design and implementation of ulmBLAS was strongly inspired by the ideas
presented in:

Field G. Van Zee and Robert A. van de Geijn. *BLIS: A Framework for Rapidly
Instantiating BLAS Functionality.* ACM Transactions on Mathematical Software,
41(3), 14:1–14:33, June 2015.
[https://doi.org/10.1145/2764454](https://doi.org/10.1145/2764454)

At the time of the initial development of ulmBLAS (mainly in 2014), I was
referring to a preprint version of this paper:
[https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf](https://www.cs.utexas.edu/~flame/pubs/blis1_toms_rev3.pdf)

---


## Citation

ulmBLAS is an independent project developed and maintained by **Michael C. Lehn** (Ulm University).  
If you use ulmBLAS in your work, please cite it as:

Michael C. Lehn. *ulmBLAS: A high performance BLAS implementation.* Ulm University, GitHub. https://github.com/michael-lehn/ulmBLAS, 2014.

### BibTeX:

```bibtex
@misc{Lehn_ulmBLAS,
  author       = {Michael C. Lehn},
  title        = {ulmBLAS: A high performance BLAS implementation},
  year         = {2014},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/michael-lehn/ulmBLAS}},
  note         = {Ulm University, Version 1.0}
}

