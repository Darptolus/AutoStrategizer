#!/usr/bin/env bash
export OMP_PROC_BIND=spread 
taskset -c 0,1,20 ./d2d_test_8.x