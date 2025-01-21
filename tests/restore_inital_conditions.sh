#!/bin/bash

#    Filename: restore_initial_conditions.sh
#    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
#    Date created: January 1, 2024
#    Description:

#    Script to restore rerun tests
#    Pass the test case as an argument, e.g.:
#    ./restore_rerun_files.sh 3_RA/A_Triangular_Shen

# (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero,
# Benjamin Hohermuth, David F. Vetsch, Robert M. Boes,
# D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)
# This software is released under the the GNU General Public License v3.0.
# https://https://opensource.org/license/gpl-3-0


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
	  python ../stsg_ssg.py -r timeseries -n 1 $group/$test'/run'
	  cp $group/$test'/run/flow_data.h5' $group/$test'/input/'
	  rm -r $group/$test'/run'
	fi

	if [[ "$file" == *"binary_signal.h5"* ]]; then
	  echo "Restoring binary_signal.h5"
	  rm -r $group/$test'/run'
	  mkdir $group/$test'/run'
	  cp -r $group/$test'/input/config.json' $group/$test'/run/'
	  python ../stsg_ssg.py -r all -n 1 $group/$test'/run'
	  cp $group/$test'/run/binary_signal.h5' $group/$test'/input/'
	  rm -r $group/$test'/run'
	fi
done
echo 'done!'
