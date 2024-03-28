#!/bin/bash

num_gpus=(2 3 4)

for ng in "${num_gpus[@]}"; do
    targetFile="${ng}GPUs.csv"
    overallFile="overall.csv"
    bigDir="Results"
    
    rm -r $bigDir
    rm $targetFile
    rm $overallFile
    rm average.txt
    rm times.txt


    mkdir -p $bigDir


    echo -e "Blocks, threads, runtime\n" > $targetFile 
    echo -e "Blocks, threads, avg_runtime\n" > $overallFile 
    # Define the list of thread counts
    thread_counts=(64 128 256 512)
    # 128 256 512
    # Define the range of block counts
    block_counts=(1 2 3 4)
    
   
    # Iterate through the combinations of thread and block counts
    for blocks in "${block_counts[@]}"; do
        for threads in "${thread_counts[@]}"; do

            template_info="NCCL_ALGO=Ring\nNCCL_PROTO=SIMPLE\nNCCL_NTHREADS=${threads}\nNCCL_MIN_NCHANNELS=${blocks}\nNCCL_MAX_NCHANNELS=${blocks}\nNCCL_DEBUG=INFO\nNCCL_DEBUG_FILE=nccl_${blocks}_blocks_${threads}_threads.debug"
            # Output the result or save it to a file
            echo -e $template_info > ~/.nccl.conf

            WORLD_SIZE=$ng
            export CUDA_DEVICE_MAX_CONNECTIONS=1
            DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                            --nnodes 1 \
                            --node_rank 0 \
                            --master_addr localhost \
                            --master_port 6000"

            TRAIN_DATA="../data/glue_data/MNLI/train.tsv"
            VALID_DATA="../data/glue_data/MNLI/dev_matched.tsv \
                        ../data/glue_data/MNLI/dev_mismatched.tsv"
            VOCAB_FILE=../vocab/bert-large-uncased-vocab.txt

            PRETRAINED_CHECKPOINT=checkpoints/bert_345m
            CHECKPOINT_PATH=checkpoints/bert_345m_mnli


            python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ../tasks/main.py \
                        --task MNLI \
                        --seed 1234 \
                        --train-data $TRAIN_DATA \
                        --valid-data $VALID_DATA \
                        --tokenizer-type BertWordPieceLowerCase \
                        --vocab-file $VOCAB_FILE \
                        --epochs 1 \
                        --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
                        --tensor-model-parallel-size 1 \
                        --num-layers 24 \
                        --hidden-size 1024 \
                        --num-attention-heads 16 \
                        --micro-batch-size 8 \
                        --lr 5.0e-5 \
                        --lr-decay-style linear \
                        --lr-warmup-fraction 0.065 \
                        --seq-length 512 \
                        --max-position-embeddings 512 \
                        --save-interval 500000 \
                        --save $CHECKPOINT_PATH \
                        --log-interval 1 \
                        --log-throughput \
                        --eval-interval 500 \
                        --eval-iters 5 \
                        --weight-decay 1.0e-1 \
                        --fp16 \
                        --overlap-grad-reduce \
                        --exit-interval 1\
                        --timing-log-level 1 \
                        --profile \
                        --profile-step-start 1
        
            sed "s/^/${blocks},${threads},/" times.txt >> $targetFile
            sed "s/^/${blocks},${threads},/" average.txt >> $overallFile

            dir=${blocks}_blocks_${threads}_threads
            mkdir -p $dir 
            
            mv *.debug $dir/
            mv times.txt $dir/
            mv average.txt $dir/
            mv $dir $bigDir/

        done
    done
    
    sed '/^$/d' $targetFile > fixed$targetFile
    mv $targetFile $bigDir/
    mv fixed$targetFile $bigDir/
    sed '/^$/d' overall.csv > fixedOverall.csv
    mv overall.csv $bigDir/
    mv fixedOverall.csv $bigDir/
    mv $bigDir ${ng}GPUResults
    
    # GPU done
done

