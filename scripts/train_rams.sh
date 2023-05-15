if [ $# == 0 ] 
then
    SEED=41
    LR=2e-5
    BatchSize=4
else
    SEED=$1
    LR=$2
fi

work_path=exp/rams/$SEED
mkdir -p $work_path

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --dataset_type=rams \
    --context_representation=decoder \
    --model_name_or_path=roberta-large \
    --role_path=./data/dset_meta/description_rams.csv \
    --prompt_path=./data/prompts/prompts_rams_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --batch_size=$BatchSize \
    --max_steps=10000 \
    --max_enc_seq_length 500 \
    --max_dec_seq_length 200 \
    --window_size 260 \
    --warmup_steps 0.1 \
    --bipartite \
    --num_prompt_pos 10