#!/bin/bash
#SBATCH --job-name=exp1-encoder
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jianxiang.he/Project/Visual-Subtitle-Integration/nlp805/experiments/exp1_encoder/logs/slurm-%j.out
#
# Experiment 1: does a different subtitle encoder find the keyframes more
# accurately than VSI's default mpnet?
#
# Three encoders, one GPU each, run in parallel in this one job. Everything
# except --scorer is identical, and identical to the mpnet baseline in
# exp2_query (mpnet_full), so all runs are paired question by question.
#
#   sbatch nlp805/experiments/exp1_encoder/run.sbatch

VARIANTS=(
  "minilm  --scorer minilm"
  "bge     --scorer bge"
  "bm25    --scorer bm25"
)
# Shared settings: original VSI subtitle branch (full question as query, no
# subtitle offset fix), text_weight 0.3, soft threshold 0.5, top-8 frames.
# --grid_fix only stops the 4x4 grid from crashing on clips < 16 s.
COMMON="--query full --text_weight 0.3 --theta 0.5 --search_nframes 8 --seed 0 --qid_seed --grid_fix"

REPO=/home/jianxiang.he/Project/Visual-Subtitle-Integration
EXP=$REPO/nlp805/experiments/exp1_encoder
TMP=$EXP/outputs/score_history
PY=/home/jianxiang.he/.conda/envs/vsi/bin/python
export HF_HOME=/home/jianxiang.he/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$REPO

cd $REPO
mkdir -p $EXP/outputs $EXP/logs $EXP/results $TMP
echo "job=$SLURM_JOB_ID host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES start=$(date '+%F %T')"
NGPU=$($PY -c "import torch; print(torch.cuda.device_count())")
[ "$NGPU" -ge ${#VARIANTS[@]} ] || { echo "only $NGPU GPUs visible, need ${#VARIANTS[@]}"; exit 1; }
$PY -m pip freeze > $EXP/logs/pip_freeze.txt

for i in "${!VARIANTS[@]}"; do
  (
    set -- ${VARIANTS[$i]}
    NAME=$1; shift
    CMD="$PY nlp805/run_fused_search.py --name $NAME --out_dir $EXP/outputs --tmp_dir $TMP/$NAME --device cuda:$i $* $COMMON"
    echo "[$(date '+%F %T')] $CMD" | tee -a $EXP/logs/commands.txt
    $CMD > $EXP/logs/$NAME.log 2>&1
    echo "[$(date '+%F %T')] $NAME exit=$?" | tee -a $EXP/logs/commands.txt
  ) &
done
wait

# Compare with the mpnet baseline if experiment 2 has already produced it.
BASE=$REPO/nlp805/experiments/exp2_query/outputs/mpnet_full.json
if [ -f $BASE ]; then
  $PY nlp805/eval_fused.py --base mpnet --runs mpnet=$BASE \
      minilm=$EXP/outputs/minilm.json bge=$EXP/outputs/bge.json bm25=$EXP/outputs/bm25.json \
      | tee $EXP/results/eval_vs_mpnet.txt
else
  echo "exp2 mpnet_full.json not found yet; run the comparison after exp2 finishes."
fi
echo "end=$(date '+%F %T')"
