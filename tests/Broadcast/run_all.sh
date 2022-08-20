#!/usr/bin/env bash

DIR_NAME="results_1"
mkdir $DIR_NAME

for i in 256 2560 25600 262144 2621440 26214400 268435456
do 
  export ARR_SZ=$i
  for j in $(seq 0 12)
  do
    # echo "ARR_SZ = $ARR_SZ"
    ./broadcast_20.x>>./$DIR_NAME/out_test_$i.o  2>&1 
  done
done

# export ARR_SZ=100; ./broadcast_20.x