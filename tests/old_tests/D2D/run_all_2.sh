#!/usr/bin/env bash

DIR_NAME="results"
if [ ! -d "$DIR_NAME" ]; then
  mkdir $DIR_NAME
fi

op_name=D2D
current_time=$(date "+%Y_%m_%d-%H%M%S")
file_name=$op_name\_$current_time.txt
all_files=""

echo "$file_name"
touch "$DIR_NAME/$file_name"

for i in 15360 153600 1572864 15728640 157286400 1610612736
do 
  export ARR_SZ=$i
  for j in $(seq 0 11)
  do
    # echo "ARR_SZ = $ARR_SZ"
    export OMP_PROC_BIND=spread 
    taskset -c 0,1,22,23 ./d2d_test_11>>./$DIR_NAME/out_test_$i.o  2>&1; wait
  done
  all_files=$all_files" "$DIR_NAME/out_test_$i.o
done

paste -d"\t" $all_files > $DIR_NAME/$file_name

rm results/out_test_*