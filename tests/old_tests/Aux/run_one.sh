#!/usr/bin/env bash
export OMP_PROC_BIND=spread 
taskset -c 0,1,20 ./auto_s_v0.x