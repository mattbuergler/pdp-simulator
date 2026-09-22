#!/bin/bash

NODE=$1
# Directory where log files will be stored
LOG_DIR="$NODE/logs_eval"
mkdir -p "$LOG_DIR"

# File to monitor the status of all scripts
STATUS_LOG="$LOG_DIR/status.log"

# The batch numbers
declare -a batch_numbers=("z001" "z005" "z010" "z014" "z019" "z023" "z028" "z032" "z036" "z041" "z045" "z050" "z054" "z059" "z063")

# Function to run a Python script with error handling
run_python_script() {
    batch_number=$1
    if [[ "$my_string" == *"four_tip"* ]]; then
        pipenv run python -u run_pdp_sim_eval_full.py -m "$NODE/${batch_number}" -r False -p True -o > "$LOG_DIR/log_${batch_number}.txt" 2>&1
    else
        pipenv run python -u run_pdp_sim_eval_full.py -m "$NODE/${batch_number}" -r False -o > "$LOG_DIR/log_${batch_number}.txt" 2>&1
    fi

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