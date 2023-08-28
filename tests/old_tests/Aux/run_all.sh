#!/usr/bin/env bash

DIR_NAME="results_d2d_1"
mkdir $DIR_NAME

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
    ./d2d_test_2.x>>./$DIR_NAME/out_test_$i.o  2>&1; wait
    # ToDo: Add Paste
  done
done

