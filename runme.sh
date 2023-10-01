#!/usr/bin/env bash
rm -rf build
mkdir build

cmake -S .                            \
      -B build                        \
      -DCMAKE_CXX_COMPILER=clang++    \
      -DVERBOSE_MODE=1
make -C build -j

# Copy topology files
cp tests/topo_* build/tests

# Copy running scripts
cp tests/old_tests/D2D/run_all.sh build/tests/old_tests/D2D/
cp tests/old_tests/D2D/run_all_2.sh build/tests/old_tests/D2D/
cp tests/old_tests/Broadcast/run_all.sh build/tests/old_tests/Broadcast/

# Copy python scripts
cp tests/old_tests/D2D/D2D_plot_a.ipynb build/tests/old_tests/D2D/
cp tests/old_tests/D2D/D2D_plot_b.ipynb build/tests/old_tests/D2D/
cp tests/old_tests/Broadcast/Broadcast_plot.ipynb build/tests/old_tests/Broadcast/