#!/bin/bash
# OutputFile="$1"
InputFile="/home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5"
OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/RUNTIME_EVAL/defaultconfig_event1004.npz"
LogFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/RUNTIME_EVAL/log_defaultconfig_event1004.log"
uv run tred -c config.yaml -l $LogFile fullsim \
   -i $InputFile \
   -o $OutputFile
