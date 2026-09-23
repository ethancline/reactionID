#!/usr/bin/env bash
# Everything after `reactionID.py train`, in order: export both models, deploy them
# into the cooker, cook the validation file with each, check cooker/Python parity,
# run the cooker's truth comparison, and build the report and the review deck.
#
#   ./review_pipeline.sh            # from the reactionID directory
#
# The MLP checkpoint must already exist (reactionID.py train --holdout-frac ...);
# export_onnx.py rebuilds its holdout from the split recorded in it.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
MUSE=$(cd ../muse && pwd)
PY=${PY:-$HERE/rID/bin/python}

# The MUSE cooker, not Homebrew's unrelated binary of the same name, with its
# libraries. Exported here rather than inherited: macOS strips DYLD_* when it
# launches a system binary (nohup, env, ...), and the cooker needs them.
export DYLD_LIBRARY_PATH=$HOME/.muse/arm64/lib:${ROOTSYS:-/Users/ethancline/packages/root.build}/lib:${DYLD_LIBRARY_PATH:-}
COOKER=$HOME/.muse/arm64/bin/cooker

TRAIN=data/train.csv,data_15Apr25/train.csv
VALID_TAG=mc17606_210MeV_LH2_2
B=$HERE/cooked/$VALID_TAG
RAW=$HERE/mc_merged/$VALID_TAG.root
mkdir -p logs results cooker_plots

echo "== export both models, thresholds from the training holdout"
$PY export_onnx.py --all --verify --train $TRAIN 2>&1 | tee logs/export.log
grep -q "ALL CHECKS PASS" logs/export.log

echo "== deploy into the cooker"
cp model/decay_gbdt.onnx model/decay_mlp.onnx "$MUSE/src/plugins/analysis/ReactionID/models/"
(cd "$MUSE/build" && make install >/dev/null)

echo "== cook the validation file with each model"
INPUTS=${B}_Vertex.root:${B}_STT_tracked.root:${B}_GEM_tracked.root:${B}_BH.root:${B}_SPS.root:${B}_VETO.root:${B}_BM.root
cd "$MUSE"
$COOKER recipes/ReactionID/ReactionID.xml "$INPUTS" ${B}_RID.root \
    -c ReactionID:setMomentum:210 -c ReactionID:setTargetPosition:0 > $HERE/logs/cook_rid_gbdt.log 2>&1
$COOKER recipes/ReactionID/ReactionID.xml "$INPUTS" ${B}_RID_mlp.root \
    -c ReactionID:setMomentum:210 -c ReactionID:setTargetPosition:0 \
    -c 'ReactionID:setModel:"decay_mlp.onnx"' > $HERE/logs/cook_rid_mlp.log 2>&1
grep "ReactionID: loaded" $HERE/logs/cook_rid_gbdt.log $HERE/logs/cook_rid_mlp.log
cd "$HERE"

echo "== cooker vs Python, event by event"
$PY verify_inference.py --rid ${B}_RID.root --csv ${B}_features.csv --model model/decay_gbdt.onnx --json results/parity_gbdt.json
$PY verify_inference.py --rid ${B}_RID_mlp.root --csv ${B}_features.csv --model model/decay_mlp.onnx --json results/parity_mlp.json

echo "== the cooker's truth comparison, beta alignment from training file 1"
ALIGN=()
while IFS= read -r line; do ALIGN+=(-c "$line"); done < <(grep -v '^#' mc17606_beta_alignment.txt)
cd "$MUSE"
$COOKER recipes/ReactionID/ReactionID_Plots_MC.xml \
    $RAW:${B}_Vertex.root:${B}_RID.root:${B}_STT_tracked.root:${B}_GEM_tracked.root:${B}_BH.root:${B}_SPS.root \
    $HERE/cooker_plots/valid_file2.root -c ReactionID:setMomentum:210 -c ReactionID:setTargetPosition:0 "${ALIGN[@]}" \
    > $HERE/logs/plots_valid_file2.log 2>&1
cd "$HERE"
grep "ReactionID truth comparison\|full reconstruction only" logs/plots_valid_file2.log

echo "== report, audit, deck"
$PY make_report.py --plots 2>&1 | tee logs/make_report.log
$PY audit_data.py --rid ${B}_RID.root 2>&1 | tee logs/audit.log
$PY make_presentation.py
