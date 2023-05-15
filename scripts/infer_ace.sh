if [ $# == 0 ] 
then
    SEED=43
    LR=2e-5
    BatchSize=8
else
    SEED=$1
    LR=$2
fi

work_path=Infer/ace
mkdir -p $work_path

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --dataset_type=ace_eeqa \
    --context_representation=decoder \
    --model_name_or_path=roberta-large \
    --inference_model_path=./exp/ace/43/checkpoint \
    --role_path=./data/dset_meta/description_ace.csv \
    --prompt_path=./data/prompts/prompts_ace_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --batch_size=$BatchSize \
    --max_steps=10000 \
    --max_enc_seq_length 200 \
    --max_dec_seq_length 250 \
    --window_size 250 \
    --bipartite \
    --inference_only \
    --single