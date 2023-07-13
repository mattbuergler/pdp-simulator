#!/bin/bash
#
# script to restore rerun tests
#
# pass the test case as an argument
# e.g.. ./restore_rerun_files.sh 4_restart/F_rerun
#
# by SJP,2018

directory=$1
subdirectories=()

# Find directories named "testdef" recursively
while IFS= read -r -d '' file; do
    subdirectories+=("$(dirname "$file")")
done < <(find "$directory" -type f -name "testdef" -print0)

# Output the list of subdirectories
for subdir in "${subdirectories[@]}"; do
    echo "$subdir"
done

# Access the sorted test directories
for subdir in "${subdirectories[@]}"; do
	echo "restoring flow_data.h5 or binary_signal.h5 for test <$subdir>"
	group="$(cut -d'/' -f1 <<< $subdir)"
	test="$(cut -d'/' -f2 <<< $subdir)"
	file=$(find $group'/'$test'/input' -type f -name '*.h5')
	if [[ "$file" == *"flow_data.h5"* ]]; then
	  echo "Restoring flow_data.h5"
	  rm -r $group/$test'/run'
	  mkdir $group/$test'/run'
	  cp -r $group/$test'/input/config.json' $group/$test'/run/'
	  python ../sbg.py -r timeseries -n 1 $group/$test'/run'
	  cp $group/$test'/run/flow_data.h5' $group/$test'/input/'
	  rm -r $group/$test'/run'
	fi

	if [[ "$file" == *"binary_signal.h5"* ]]; then
	  echo "Restoring binary_signal.h5"
	  rm -r $group/$test'/run'
	  mkdir $group/$test'/run'
	  cp -r $group/$test'/input/config.json' $group/$test'/run/'
	  python ../sbg.py -r all -n 1 $group/$test'/run'
	  cp $group/$test'/run/binary_signal.h5' $group/$test'/input/'
	  rm -r $group/$test'/run'
	fi
done
echo 'done!'
