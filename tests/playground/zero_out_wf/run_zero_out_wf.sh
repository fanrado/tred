#!/bin/bash
# OutputFile="$1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OutputFile="/home/rrazakami/work/ND-LAr/tredBenchmark_update_Dec17_2025/OUTPUT_EVAL/MEMORY_EVAL/default_config_event1003.npz"
OutputLog="/home/rrazakami/work/ND-LAr/tredBenchmark_update_Dec17_2025/OUTPUT_EVAL/MEMORY_EVAL/mem_benchmark_differentBatchsizes_event1003.log"
InputFile="/home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5"
# InputFile="/home/rrazakami/work/ND-LAr/data_tred/MR5_2x2_single_particles/segments_pid211.hdf5"
uv run tred -c config.yaml -l $OutputLog fullsim \
   -i $InputFile \
   -o $OutputFile
