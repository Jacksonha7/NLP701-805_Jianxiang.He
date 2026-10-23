#!/bin/bash
#SBATCH --job-name=exp2-query
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jianxiang.he/Project/Visual-Subtitle-Integration/nlp805/experiments/exp2_query/logs/slurm-%j.out
#
# Experiment 2: with VSI's own mpnet matching, is it more accurate to match the
# subtitle lines against only the subtitle text quoted in the question than
# against the whole question?
#
# 207 of the 220 questions quote a subtitle line; the other 13 fall back to the
# whole question in the quote run. Everything except --query is identical.
#
#   sbatch nlp805/experiments/exp2_query/run.sbatch

VARIANTS=(
  "mpnet_full   --query full"
  "mpnet_quote  --query quote"
)
# Shared settings: original VSI subtitle branch (mpnet, no subtitle offset
# fix), text_weight 0.3, soft threshold 0.5, top-8 frames. Same as exp1.
# --grid_fix only stops the 4x4 grid from crashing on clips < 16 s.
COMMON="--scorer mpnet --text_weight 0.3 --theta 0.5 --search_nframes 8 --seed 0 --qid_seed --grid_fix"

REPO=/home/jianxiang.he/Project/Visual-Subtitle-Integration
EXP=$REPO/nlp805/experiments/exp2_query
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

$PY nlp805/eval_fused.py --base mpnet_full \
    --runs mpnet_full=$EXP/outputs/mpnet_full.json mpnet_quote=$EXP/outputs/mpnet_quote.json \
    | tee $EXP/results/eval_quote_vs_full.txt

# mpnet_full is also the baseline of experiment 1; write that comparison into
# exp1's results once both experiments are done.
EXP1=$REPO/nlp805/experiments/exp1_encoder
if [ -f $EXP1/outputs/minilm.json ] && [ -f $EXP1/outputs/bge.json ] && [ -f $EXP1/outputs/bm25.json ]; then
  $PY nlp805/eval_fused.py --base mpnet --runs mpnet=$EXP/outputs/mpnet_full.json \
      minilm=$EXP1/outputs/minilm.json bge=$EXP1/outputs/bge.json bm25=$EXP1/outputs/bm25.json \
      | tee $EXP1/results/eval_vs_mpnet.txt
fi
echo "end=$(date '+%F %T')"
