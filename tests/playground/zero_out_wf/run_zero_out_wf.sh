#!/bin/bash
# OutputFile="$1"
OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/default_config_event1002.npz"
OutputLog="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/log_default_config_event1002.log"
InputFile="/home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5"
# InputFile="/home/rrazakami/work/ND-LAr/data_tred/MR5_2x2_single_particles/segments_pid211.hdf5"
uv run tred -c config.yaml -l $OutputLog fullsim \
   -i $InputFile \
   -o $OutputFile
