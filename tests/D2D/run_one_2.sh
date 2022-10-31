#!/usr/bin/env bash
export OMP_PROC_BIND=spread 
taskset -c 0,1,22,23 ./d2d_test_11.x