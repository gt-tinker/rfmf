# Multi-chain (throughput) CUDA Gumbel-competition annealer for Max-Cut.
#
# Everything is built by nvcc in one shot: philox.cuh is shared verbatim between
# the device kernels and the host, so keeping one compiler avoids two ABIs and
# any chance of the two drifting.
#
#   make          -> ./main
#   make clean
#
# The artifact is built and tuned for ONE target: the Tesla V100 (sm_70).

SHELL := /bin/bash

# --- toolchain -------------------------------------------------------------
#
# The measured configuration for this artifact is:
#
#   nvcc  release 12.9, V12.9.41   (CUDA 12.9)
#   g++   11.4.0                   (Ubuntu 11.4.0-1ubuntu1~22.04.3)
#
# Both are checked below and a mismatch is a hard error, so a reviewer knows
# immediately that they are not reproducing on the tested stack.  To build
# anyway on a nearby CUDA 12.x / g++ 11.x-adjacent toolchain:
#
#   make ALLOW_TOOLCHAIN_MISMATCH=1
#
# CUDA 13.x will NOT work at all: it dropped Volta/sm_70.
#
# nvcc links the CUDA runtime statically by default (--cudart=static), so the
# binary carries no libcudart.so dependency and needs no LD_LIBRARY_PATH.

NVCC  ?= nvcc
CCBIN ?= g++

REQUIRED_CUDA := 12.9
REQUIRED_GCC  := 11

# The gates below are $(error)s, which fire while the makefile is being READ --
# so they would also abort goals that need no compiler at all.  Skip them when
# every requested goal is compiler-free, or `make clean` would be unusable on a
# machine that cannot build.
COMPILER_FREE_GOALS := clean
ifeq ($(strip $(MAKECMDGOALS)),)
  NEEDS_TOOLCHAIN := 1
else ifeq ($(strip $(filter-out $(COMPILER_FREE_GOALS),$(MAKECMDGOALS))),)
  NEEDS_TOOLCHAIN := 0
else
  NEEDS_TOOLCHAIN := 1
endif

ifeq ($(NEEDS_TOOLCHAIN),1)

NVCC_OK := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes)
ifneq ($(NVCC_OK),yes)
  $(error nvcc not found (tried '$(NVCC)'). Install CUDA $(REQUIRED_CUDA), put nvcc on \
PATH, or pass `make NVCC=/path/to/nvcc`)
endif

CCBIN_OK := $(shell command -v $(CCBIN) >/dev/null 2>&1 && echo yes)
ifneq ($(CCBIN_OK),yes)
  $(error host compiler not found (tried '$(CCBIN)'). Pass `make CCBIN=/path/to/g++`)
endif

# "Cuda compilation tools, release 12.9, V12.9.41" -> 12.9
CUDA_VERSION := $(shell $(NVCC) --version 2>/dev/null | \
                  sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
GCC_VERSION  := $(shell $(CCBIN) -dumpversion 2>/dev/null | cut -d. -f1)

# CUDA >= 13 is NOT a version preference: it removed sm_70 entirely, so it can
# never target a V100.  This check therefore stands even under the escape hatch;
# letting it through would only buy an opaque `nvcc fatal` a minute later.
ifeq ($(firstword $(subst ., ,$(CUDA_VERSION))),13)
  $(error $(NVCC) is CUDA $(CUDA_VERSION), which CANNOT build this artifact at all: \
CUDA 13.x dropped sm_70/Volta, so it cannot target the V100. Install CUDA \
$(REQUIRED_CUDA) and pass `make NVCC=/path/to/nvcc`. ALLOW_TOOLCHAIN_MISMATCH=1 does \
NOT help here)
endif

ifneq ($(ALLOW_TOOLCHAIN_MISMATCH),1)
  ifeq ($(CUDA_VERSION),)
    $(error could not read a version out of `$(NVCC) --version` -- is '$(NVCC)' a \
working nvcc? This artifact was measured with CUDA $(REQUIRED_CUDA); pass \
`make NVCC=/path/to/nvcc`, or `make ALLOW_TOOLCHAIN_MISMATCH=1` to try anyway)
  endif
  ifneq ($(CUDA_VERSION),$(REQUIRED_CUDA))
    $(error this artifact was measured with CUDA $(REQUIRED_CUDA) but $(NVCC) is \
CUDA $(CUDA_VERSION). To build anyway on another CUDA 12.x: \
make ALLOW_TOOLCHAIN_MISMATCH=1)
  endif
  ifneq ($(GCC_VERSION),$(REQUIRED_GCC))
    $(error this artifact was measured with g++ $(REQUIRED_GCC).x but $(CCBIN) is \
$(GCC_VERSION).x. To build anyway: make ALLOW_TOOLCHAIN_MISMATCH=1)
  endif
else
  $(info NOTE: toolchain check bypassed -- building with CUDA $(CUDA_VERSION), \
g++ $(GCC_VERSION).x instead of the measured CUDA $(REQUIRED_CUDA) / g++ $(REQUIRED_GCC).x)
endif

endif  # NEEDS_TOOLCHAIN

# --- target architecture ---------------------------------------------------
#
# sm_70 (Volta / Tesla V100) only.  No PTX fallback and no other architecture:
# every tuned constant in src/tuning.hpp was measured on this one card, so a
# binary that silently ran elsewhere would report untuned numbers.
GENCODE = -gencode arch=compute_70,code=sm_70

# -Wno-deprecated-gpu-targets silences CUDA 12.9's "architectures prior to
# sm_75 will be removed" note, which otherwise fires on every translation unit
# and buries the real -Wall/-Wextra host warnings.
# -lineinfo keeps source line mapping in the cubin so nsight-compute/nsys can
# attribute the kernel's stalls to lines of anneal_kernels.cu.  It does not
# change codegen.
# --expt-relaxed-constexpr is defensive only: the Gumbel LUT the device reads
# is kGumbelCoarseDevG, a plain __device__ array, so nothing currently calls a
# constexpr host function from device code and the build succeeds without this
# flag.  Kept so that adding such a call fails at the call site rather than
# with an unrelated nvcc diagnostic.
NVCCFLAGS = -std=c++17 -O2 -Isrc -lineinfo --expt-relaxed-constexpr \
            -Wno-deprecated-gpu-targets \
            -ccbin $(CCBIN) \
            -Xcompiler "-Wall,-Wextra,-O2"

TARGET = main

SRCS = main.cu \
       src/host_common.cpp src/read_gset.cpp src/graph.cpp src/max_cut.cpp \
       src/anneal_kernels.cu

HDRS = $(wildcard src/*.h) $(wildcard src/*.hpp) $(wildcard src/*.cuh)

.PHONY: all clean

all: $(TARGET)

# One-shot compile: nvcc sees every translation unit at once, so there are no
# .o files to keep in sync.
$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
