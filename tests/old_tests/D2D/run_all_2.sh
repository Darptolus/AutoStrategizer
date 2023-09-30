#!/usr/bin/env bash

DIR_NAME="results_d2d_smx_0"
mkdir $DIR_NAME

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
file_name=$op_name\_$current_time

# 15360; // 60 Kb
# 153600; // 600 Kb
# 1572864; // 6 Mb
# 15728640; // 60 Mb
# 157286400; // 600 Mb
# 1610612736; // 6 Gb

for i in 15360 153600 1572864 15728640 157286400 1610612736
do 
  export ARR_SZ=$i
  for j in $(seq 0 11)
  do
    # echo "ARR_SZ = $ARR_SZ"
    export OMP_PROC_BIND=spread 
    taskset -c 0,1,22,23 ./d2d_test_7.x>>./$DIR_NAME/out_test_$i.o  2>&1; wait
    # ToDo: Add Paste
    # paste -d"\t"  out_test_256.o out_test_2560.o out_test_25600.o out_test_262144.o out_test_2621440.o out_test_26214400.o out_test_209715200.o out_test_235929600.o>test.o 2>&1
  done
done

