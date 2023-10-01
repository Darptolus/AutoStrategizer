#!/usr/bin/env bash

DIR_NAME="results"
if [ ! -d "$DIR_NAME" ]; then
  mkdir $DIR_NAME
fi

op_name=broadcast
current_time=$(date "+%Y_%m_%d-%H%M%S")
file_name=$op_name\_$current_time.txt
all_files=""

echo "$file_name"
touch "$DIR_NAME/$file_name"

# 256; // 1 Kb
# 2560; // 10 Kb
# 25600; // 100 Kb
# 262144; // 1 Mb
# 2621440; // 10 Mb
# 26214400; // 100 Mb
# 268435456; // 1 Gb
# 1342177280; // 5 Gb

for i in 256 2560 25600 262144 2621440 26214400 268435456 1342177280
do 
  export ARR_SZ=$i
  for j in $(seq 0 11)
  do
    # echo "ARR_SZ = $ARR_SZ"
    ./broadcast_22>>./$DIR_NAME/out_test_$i.o  2>&1; wait
  done
  all_files=$all_files" "$DIR_NAME/out_test_$i.o
done

paste -d"\t" $all_files > $DIR_NAME/$file_name

rm results/out_test_*

# export ARR_SZ=100; ./broadcast_20.x

# for i in 10