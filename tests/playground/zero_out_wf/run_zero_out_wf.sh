#!/bin/bash
# OutputFile="$1"
OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/10x10_partitions_afterconversion_.npz"
OutputLog="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/log_10x10_partitions_afterconversion_.log"
InputFile="/home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5"
# InputFile="/home/rrazakami/work/ND-LAr/data_tred/MR5_2x2_single_particles/segments_pid211.hdf5"
uv run tred -c config.yaml -l $OutputLog fullsim \
   -i $InputFile \
   -o $OutputFile
