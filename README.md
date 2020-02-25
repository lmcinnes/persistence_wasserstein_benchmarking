# Persistence Wasserstein Benchmarking Tools

Tools for benchmarking implementations of Wasserstein-Kantorovich distance between persistence diagrams.

This includes Hera, and a new binary for all pairs distances, plus python code for Wasserstein distance and mock diagram generation for benchmarking purposes.

To get started

```bash
$ git clone https://github.com/lmcinnes/persistence_wasserstein_benchmarking
$ cd persistence_wasserstein_benchmarking
$ cmake hera/geom_matching/wasserstein/
$ make
$ python generate_diagrams.py -n 50 -N 100
$
$ time ./wasserstein_dist_all_pairs -p 2 -q 1 -o hera_result_50.txt data_50/*
$ time python python_wasserstein_all_pairs.py -p 2 -q 1 -o py_result_50.txt data_50/*
```

## Requirements

The combination of Hera requirements and python + optimal transport requirements is, at a minimum:

 - Cmake
 - Boost
 - POT
 - Scikit-learn
 - Numpy
