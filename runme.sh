#!/usr/bin/env bash
cmake -S .                            \
      -B build                        \
      -DCMAKE_CXX_COMPILER=clang++    \
      -DVERBOSE_MODE=1
make -C build