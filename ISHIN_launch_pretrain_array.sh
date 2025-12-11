set -e

echo "Setting up SLURM array job..."
echo ""

# Make sure the SLURM script is executable
chmod +x ISHIN_pretrain_array.sh

# Create logs and output directories
mkdir -p logs
mkdir -p out

eval "$(conda shell.bash hook)"
conda activate trp

# Submit the array job
echo "Submitting SLURM array job..."
job_id=$(sbatch ISHIN_pretrain_array.sh | awk '{print $4}')

# Create output directory for this job
mkdir -p out/${job_id}

echo "Array job submitted with ID: $job_id"
echo ""
echo "To monitor the jobs:"
echo "  squeue -u \$USER"
echo "  squeue -j $job_id"
echo ""
echo "To cancel all jobs in the array:"
echo "  scancel $job_id"
echo ""
echo "To check job status:"
echo "  sacct -j $job_id --format=JobID,JobName,State,ExitCode,Start,End"
echo ""
echo "Log files will be written to:"
echo "  out/${job_id}/gtrmarr_${job_id}_*.out"
echo "  out/${job_id}/gtrmarr_${job_id}_*.err"
echo ""
echo "To view real-time output of a specific task (e.g., task 1):"
echo "  tail -f out/${job_id}/gtrmarr_${job_id}_1.out"