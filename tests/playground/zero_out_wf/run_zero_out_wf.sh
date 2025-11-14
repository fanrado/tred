#!/bin/bash
# OutputFile="$1"
# subdir="cut_on_drifttime_30_100_nocut_event1003"
# subdir="cap_on_diffspread"
subdir="November10_2025/torch_clamp_spreadT/"
OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/${subdir}/10x10_2x2x2_tdrift20.npz"
# OutputFile="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/event1004_pid211_segment233928.npz"
# OutputFile="debug/event1004_pid211_segment233928.npz"
OutputLog="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/${subdir}/10x10_2x2x2_tdrift20.log"
# OutputLog="/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/event1004_pid211_segment233928.log"
# OutputLog="debug/event1004_pid211_segment233928.log"
InputFile="/home/rrazakami/work/ND-LAr/data_tred/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5"
# InputFile="/home/rrazakami/work/ND-LAr/data_tred/event1004_pid211_segment233928.hdf5"
uv run tred -c config.yaml -l $OutputLog fullsim \
   -i $InputFile \
   -o $OutputFile
