#!/bin/bash
# OutputFile="$1"
OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/defaultconfig_event1004.npz"
OutputLog="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/log_graph_effq.log"
uv run tred -c config.yaml -l $OutputLog fullsim \
   -i /home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
   -o $OutputFile
