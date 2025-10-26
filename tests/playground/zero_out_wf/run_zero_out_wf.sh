#!/bin/bash
# OutputFile="$1"
# OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/NDlar_10x10_partitions_2x2x2.npz"
OutputFile="/home/rrazakami/work/ND-LAr/starting_over/tred_debug_rado_repo/tests/playground/zero_out_wf/output/MicroProdN1p1_NDLAr_1E18_RHC_event1003_9x9FR.npz"
# OutputLog="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/log_NDlar_10x10_partitions_2x2x2.log"
OutputLog="/home/rrazakami/work/ND-LAr/starting_over/tred_debug_rado_repo/tests/playground/zero_out_wf/output/MicroProdN1p1_NDLAr_1E18_RHC_event1003_9X9FR.log"
InputFile="/home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5"
# InputFile="/home/rrazakami/work/ND-LAr/data_tred/event1004_pid211_segment233928.hdf5"
uv run tred -c config.yaml -l $OutputLog fullsim \
   -i $InputFile \
   -o $OutputFile
