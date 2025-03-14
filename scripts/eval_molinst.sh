#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=false
cd ..

# Evaluate EvoLlama (ProteinMPNN+ESM-2) on the protein understanding tasks.
weights_dir="/path/to/EvoLlama"

python src/eval/eval_molinst.py \
--structure_encoder_path "${weights_dir}/structure_encoder_weights.bin" \
--structure_encoder_name "ProteinMPNN" \
--structure_embedding_dim 128 \
--sequence_encoder_path "${weights_dir}/sequence_encoder" \
--sequence-embedding_dim 1280 \
--llm_path "/path/to/llm" \
--llm_embedding_dim 4096 \
--projection_path "${weights_dir}/projection_weights.bin" \
--projection_fusion True \
--task_name all \
--dataset_dir "/path/to/test_set" \
--pdb_dir "/path/to/pdb_dir" \
--dataset_type test \
--save_result True \
--save_dir "/path/to/save_dir" \
--use_structure True \
--use_sequence True \
--use_lora False \
--batch_size 1