#!/bin/bash
# OutputFile="$1"
OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/CHECK_OUTPUT/default_config_float64_event1003.npz"
OutputLog="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/CHECK_OUTPUT/log_default_config_float64_event1003.log"
InputFile="/home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5"
# InputFile="/home/rrazakami/work/ND-LAr/data_tred/MR5_2x2_single_particles/segments_pid211.hdf5"
uv run tred -c config.yaml -l $OutputLog fullsim \
   -i $InputFile \
   -o $OutputFile
