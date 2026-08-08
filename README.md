# RFMF — Rejection-Free Multiple-Flip for MAX-CUT on sparse graphs

This repository contains the CUDA GPU implementations for the RFMF algorithms.

# Directory Structure
```
├── Gset
├── results
│   ├── V100-PCIE-10M
│   ├── V100-PCIE-100M
│   ├── V100-SXM2-10M
│   └── V100-SXM2-100M
├── scripts
│   ├── bench_large_gset.sh
│   ├── eval_cut.py
│   └── make_results_table.py
├── src
│   ├── anneal_gpu.cuh
│   ├── anneal_kernels.cu
│   ├── graph.cpp
│   ├── graph.hpp
│   ├── host_common.cpp
│   ├── host_common.h
│   ├── immutable_graph.hpp
│   ├── max_cut.cpp
│   ├── max_cut.hpp
│   ├── philox.cuh
│   ├── read_gset.cpp
│   ├── read_gset.hpp
│   ├── rng.hpp
│   └── tuning.hpp
└── tests
│   ├── run_all.sh
│   ├── test_chain_isolation.sh
│   ├── test_cut_consistency.sh
│   └── verify_committed_cuts.sh
├── Dockerfile
├── main.cu
├── Makefile
└── README.md
```

# Installation and Builds of Programs
## Docker Image
# Dataset Generation
Dataset is prodivde under Gset/. No generation process is needed.
# Execution of Programs


