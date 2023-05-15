if [ $# == 0 ] 
then
    SEED=44
    LR=2e-5
    BatchSize=4
else
    SEED=$1
    LR=$2
fi

work_path=Infer/mlee/
mkdir -p $work_path

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --dataset_type=MLEE \
    --context_representation=decoder \
    --model_name_or_path=roberta-large \
    --inference_model_path=./exp/mlee/44/checkpoint \
    --role_path=./data/MLEE/MLEE_role_name_mapping.json \
    --prompt_path=./data/prompts/prompts_MLEE_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --batch_size=$BatchSize \
    --max_steps=10000 \
    --max_enc_seq_length 500 \
    --max_dec_seq_length 360 \
    --window_size 250 \
    --bipartite \
    --inference_only \
    --num_event_embed 20