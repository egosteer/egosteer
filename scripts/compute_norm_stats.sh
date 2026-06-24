python -m src.workspace.compute_norm_stats \
    --config src/config/experiment/egosteer_qwen3_vl.yaml \
    --output_dir outputs/normalizer/example \
    --num_workers 16 \
    --max_total_shards 10000 \