#!/bin/bash
# MCLRec — 在 SECL AmazonReviews2014 / Yelp 数据集上运行
#
# 流程（每个 category）：
#   1) 数据转换: SECL cache -> dataset/<category>/<category>.inter（已有则跳过）
#   2) 训练: RecBole 全排序评测（Recall/MRR/NDCG/Precision @5/10/20/50）
#   3) 打印最终 best valid / test result 指标
#   已完成（log 中含 test result）的自动跳过，可安全重跑。
#
# 训练超参按 README 推荐值（use_rl=1 joint=0 lmd=0.03 beta=0.1 sim=dot）。
#
# 用法:
#   bash run_amazon_reviews.sh                       # 默认四数据集
#   bash run_amazon_reviews.sh Yelp Video_Games      # 指定数据集
# 环境变量: HINT_CACHE / GPU_ID / SEED / PYTHON_BIN / LMD / BETA 等
set -uo pipefail

DEFAULT_CATEGORIES=("Office_Products" "Grocery_and_Gourmet_Food" "Video_Games" "Yelp")
CATEGORIES=("$@")
[ ${#CATEGORIES[@]} -gt 0 ] || CATEGORIES=("${DEFAULT_CATEGORIES[@]}")

MODEL="MCLRec"
PYTHON_BIN="${PYTHON_BIN:-python}"
HINT_CACHE="${HINT_CACHE:-/root/autodl-tmp/HINT/cache/AmazonReviews2014}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2020}"
LMD="${LMD:-0.03}"
BETA="${BETA:-0.1}"
SIM="${SIM:-dot}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

for category in "${CATEGORIES[@]}"; do
    LOG_FILE="${LOG_DIR}/${MODEL}_${category}_seed_${SEED}.log"
    # Skip categories that already finished (log contains a final test result).
    if [ -f "${LOG_FILE}" ] && grep -q 'test result' "${LOG_FILE}" 2>/dev/null; then
        echo ">> [${category}] already finished, skipping (log: ${LOG_FILE})"
        echo "   Final results:"
        grep -E 'best valid|test result' "${LOG_FILE}" | tail -n 2 | sed 's/^/   /'
        echo ""
        continue
    fi
    if [ ! -f "dataset/${category}/${category}.inter" ]; then
        echo ">> converting SECL data: ${category}"
        "${PYTHON_BIN}" convert_secl2inter.py \
            --hint_cache_dir "${HINT_CACHE}" \
            --category "${category}" \
            --out_dir "./dataset"
    else
        echo ">> dataset exists, skipping conversion: ${category}"
    fi
    echo "Running ${MODEL} on ${category} with seed ${SEED} (lmd=${LMD}, beta=${BETA})..."
    echo "Logging to: ${LOG_FILE}"

    "${PYTHON_BIN}" run_seq.py \
        --model="${MODEL}" \
        --dataset="${category}" \
        --config_files="seq.yaml" \
        --gpu_id=0 \
        --seed="${SEED}" \
        --epochs=100 \
        --use_rl=1 \
        --joint=0 \
        --train_batch_size=256 \
        --lmd="${LMD}" \
        --beta="${BETA}" \
        --sim="${SIM}" \
        > "${LOG_FILE}" 2>&1

    echo ">> [${category}] final results:"
    grep -E 'best valid|test result' "${LOG_FILE}" | tail -n 2 | sed 's/^/   /'
    echo ""
done

echo "MCLRec experiments finished!"
