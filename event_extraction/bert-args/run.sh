BERT_BASE_DIR=bert-base-uncased
DATA_DIR=./data/w_event
OUTPUT_DIR=./model_save

python main.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_trian True\
    --do_eval True\
    --max_seq_length 256 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --test_file ${DATA_DIR}/test.txt \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --num_train_epochs 10 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR
