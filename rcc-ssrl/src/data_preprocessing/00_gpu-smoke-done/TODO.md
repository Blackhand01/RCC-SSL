# GPU Smoke Test Fix Plan

## Tasks
- [ ] Create conda environment 'train' on login node with PyTorch CUDA
- [ ] Update gpu_smoke_autolog.sbatch to fail-fast if env missing
- [ ] Update run_autopick.sh to remove env creation logic
- [ ] Test the updated scripts

## Information Gathered
- Current sbatch creates env in job, causing downloads on compute node (bad)
- Job 898039 is stuck creating env on compute-3-13
- Need env created on login node first
- Use miniconda3/3.13.25 module
- Install PyTorch with CUDA 12.1

## Next Steps
1. Execute env creation on login node
2. Edit sbatch script
3. Edit wrapper script
4. Run test job
