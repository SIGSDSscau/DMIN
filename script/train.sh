# zh
export CUDA_VISIBLE_DEVICES=1

python main.py \
    --seed 42 \
    --result_file_name result \
    --lang zh \
    --dscgnn_layer_num 2 \
    --gnn_layer_num 3 \
    --loss_w 296 \
    --topk 0.8 \
    --warmup_steps 350 \
    --adam_epsilon 1e-7 \
    --input_files "train_dependent_trf valid_dependent_trf test_dependent_trf"


# en
python main.py \
    --seed 42 \
    --result_file_name result \
    --lang en \
    --dscgnn_layer_num 2 \
    --gnn_layer_num 3 \
    --loss_w 296 \
    --topk 0.5 \
    --warmup_steps 400 \
    --adam_epsilon 1e-8 \
    --input_files "train_dependent_trf valid_dependent_trf test_dependent_trf"