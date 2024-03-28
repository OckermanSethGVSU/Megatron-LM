#!/bin/bash

num_gpus=(2 4)

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

            rm -r checkpoints/

            template_info="NCCL_ALGO=Ring\nNCCL_PROTO=SIMPLE\nNCCL_NTHREADS=${threads}\nNCCL_MIN_NCHANNELS=${blocks}\nNCCL_MAX_NCHANNELS=${blocks}\nNCCL_DEBUG=INFO\nNCCL_DEBUG_FILE=nccl_${blocks}_blocks_${threads}_threads.debug"
            # Output the result or save it to a file
            echo -e $template_info > ~/.nccl.conf

            GPUS_PER_NODE=$ng
            export CUDA_DEVICE_MAX_CONNECTIONS=1

            # Change for multinode config
            MASTER_ADDR=localhost
            MASTER_PORT=6000
            NNODES=1
            NODE_RANK=0
            WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

            CHECKPOINT_PATH=checkpoints/
            VOCAB_FILE=../vocab/gpt2-vocab.json
            MERGE_FILE=../vocab/gpt2-merges.txt
            DATA_PATH=../data/my-gpt2/my-gpt2_text_document

            DISTRIBUTED_ARGS="
                --nproc_per_node $GPUS_PER_NODE \
                --nnodes $NNODES \
                --node_rank $NODE_RANK \
                --master_addr $MASTER_ADDR \
                --master_port $MASTER_PORT
            "

            GPT_ARGS="
                --num-layers 24 \
                --hidden-size 1024 \
                --num-attention-heads 16 \
                --seq-length 512 \
                --max-position-embeddings 512 \
                --micro-batch-size 8 \
                --global-batch-size 72 \
                --lr 0.00015 \
                --train-iters 500000 \
                --lr-decay-iters 320000 \
                --lr-decay-style cosine \
                --min-lr 1.0e-5 \
                --weight-decay 1e-2 \
                --lr-warmup-fraction .01 \
                --clip-grad 1.0 \
                --fp16
            "

            DATA_ARGS="
                --data-path $DATA_PATH \
                --vocab-file $VOCAB_FILE \
                --merge-file $MERGE_FILE \
                --split 949,50,1
            "

            OUTPUT_ARGS="
                --log-interval 100 \
                --save-interval 10000 \
                --eval-interval 1000 \
                --eval-iters 10
            "

            torchrun $DISTRIBUTED_ARGS ../pretrain_gpt.py \
                $GPT_ARGS \
                $DATA_ARGS \
                $OUTPUT_ARGS \
                --distributed-backend nccl \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
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

