#!/bin/bash

NODE=$1
# Directory where log files will be stored
LOG_DIR="$NODE/logs"
mkdir -p "$LOG_DIR"

# File to monitor the status of all scripts
STATUS_LOG="$LOG_DIR/status.log"

# The batch numbers
declare -a batch_numbers=("0000" "0001" "0002" "0003" "0004" "0005" "0006" "0007" "0008" "0009" "0010" "0011" "0012" "0013" "0014" "0015" "0016" "0017" "0018" "0019" "0020" "0021" "0022" "0023" "0024" "0025" "0026" "0027" "0028" "0029")

# Function to run a Python script with error handling
run_python_script() {
    batch_number=$1
    pipenv run python -u run_mpd_mc.py -m "$NODE/batch_${batch_number}" -r False > "$LOG_DIR/log_${batch_number}.txt" 2>&1

    if [ $? -ne 0 ]; then
        echo "Batch ${batch_number} failed" >> "$STATUS_LOG"
    else
        echo "Batch ${batch_number} completed successfully" >> "$STATUS_LOG"
    fi
}

# Loop to start all Python scripts using the numbers in the array
for batch_number in "${batch_numbers[@]}"; do
    run_python_script $batch_number &
done

# Wait for all scripts to finish
wait

echo "All scripts have finished. Check $STATUS_LOG for status."