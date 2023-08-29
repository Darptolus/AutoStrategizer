#!/usr/bin/env bash
rm -rf build
mkdir build

cmake -S .                            \
      -B build                        \
      -DCMAKE_CXX_COMPILER=clang++    \
      -DVERBOSE_MODE=1
make -C build -j

cp tests/topo_* build/tests