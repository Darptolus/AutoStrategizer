#!/usr/bin/env bash

for i in 256 2560 25600 262144 2621440 26214400 268435456
do 
  for j in $(seq 0 10)
  do
    export ARR_SZ=$i
    # echo "ARR_SZ = $ARR_SZ"
    ./test_8.x>>./results/out_test_$i.o  2>&1 
  done
done

# export ARR_SZ=100; ./broadcast_20.x