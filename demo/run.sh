torchrun --nproc_per_node 2 \
    text_generation_tp.py \
    --model /data/yi-34b-testTPS/Yi-34B-200K \
    --max-tokens 4096 \
    --eos-token $'\n' \
    --streaming
