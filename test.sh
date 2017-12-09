for((i=0; i<31; i++))
do
    export SLURM_ARRAY_TASK_ID=$i
    bash slurm-scripts/cluster.sbatch.sh
done

