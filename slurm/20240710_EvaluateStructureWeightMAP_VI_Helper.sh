#!/bin/bash

# Directory to list files from
SOURCE_DIRECTORY=$1
# Directory to check for processed files
PROCESSED_DIRECTORY=$2
# Path to the SLURM job script template
SLURM_JOB_SCRIPT_TEMPLATE="/mnt/home/tudomlumleart/ceph/01_ChromatinEnsembleRefinement/chromatin-ensemble-refinement/scripts/slurm/20240710_EvaluateStructureWeightMAP_VI.sh"

# Check if the required arguments are provided
if [ -z "$SOURCE_DIRECTORY" ] || [ -z "$PROCESSED_DIRECTORY" ]; then
    echo "Usage: $0 <source_directory> <processed_directory>"
    exit 1
fi

# Function to count currently running/queued jobs
count_jobs() {
    squeue -u $USER | grep -c ' R\|PD '
}

# List all files in the source directory
echo "Listing files in directory: $SOURCE_DIRECTORY"
FILES=$(ls "$SOURCE_DIRECTORY"/*.pkl)

# Loop through each file
for FILE in $FILES; 
do
    # Extract the filename from the full path
    FILENAME=$(basename "$FILE")
    
    # Extract the common part of the filename
    COMMON_PART=$(echo "$FILENAME" | cut -d'_' -f1-7)

    # Construct the expected results filename
    RESULTS_FILE="${COMMON_PART}_0_results.txt"

    # Check if the results file already exists
    if [ -f "$PROCESSED_DIRECTORY/$RESULTS_FILE" ]; then
        echo "File $FILENAME has already been processed as $RESULTS_FILE. Skipping."
    else

        # Submit the job 100 times with random second argument
        for i in {1..100}; 
        do
            
            # Generate a random number between 0 and 999
            RANDOM_NUMBER=$(shuf -i 0-999 -n 1)
       
    	    # Submit the job and check the number of jobs
            while [ $(count_jobs) -ge 30 ]; do
           	    echo "Waiting for available slots..."
                sleep 10
            done

            echo "Submitting job for file: $FILENAME with random number: $RANDOM_NUMBER" 
            # Submit the job using the SLURM job script template, passing the file and the random number as arguments
            sbatch $SLURM_JOB_SCRIPT_TEMPLATE "$FILE" "$RANDOM_NUMBER"
        done
    fi
done
