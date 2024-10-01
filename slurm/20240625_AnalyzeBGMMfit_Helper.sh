#!/bin/bash

# Directory to list files from
SOURCE_DIRECTORY=$1
# Directory to check for processed files
PROCESSED_DIRECTORY=$2
# Path to the SLURM job script template
SLURM_JOB_SCRIPT_TEMPLATE=/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/slurm/20240625_AnalyzeBGMMfit.sh


# Function to count currently running/queued jobs
count_jobs() {
    squeue -u $USER | grep -c ' R\|PD '
}

# List all files in the source directory
echo "Listing files in directory: $SOURCE_DIRECTORY"
FILES=$(ls "$SOURCE_DIRECTORY"/*.pkl)

# Loop through each file
for FILE in $FILES; do
    # Extract the filename from the full path
    FILENAME=$(basename "$FILE")

    # Check if the results directory already exists
    RESULTS_DIR="$PROCESSED_DIRECTORY/$FILENAME"

    if [ -d "$RESULTS_DIR" ]; then
        echo "Directory $RESULTS_DIR already exists for file $FILENAME. Skipping."
    else

        # Submit the job and check the number of jobs
        while [ $(count_jobs) -ge 30 ]; do
            echo "Waiting for available slots..."
            sleep 10
        done

        echo "Submitting job for file: $FILENAME" 
        # Submit the job using the SLURM job script template, passing the file
        sbatch $SLURM_JOB_SCRIPT_TEMPLATE "$FILE" 
        
    fi
done
