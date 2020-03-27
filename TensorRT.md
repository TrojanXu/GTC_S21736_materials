Using nvcr.io/nvidia/tensorflow:20.02-tf1-py3 
Latest BERT TRT github: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/trt  
**But we will use previous github code**: https://github.com/NVIDIA/TensorRT/tree/release/6.0/demo/BERT

1. Follow https://github.com/NVIDIA/TensorRT/tree/release/6.0/demo/BERT#example-workflow to download and build the binary
2. In Step 4 https://github.com/NVIDIA/TensorRT/tree/release/6.0/demo/BERT#4-build-and-run-the-example
    ```bash
    build/sample_bert -d $WEIGHT_PATH -d $OUTPUT_PATH --fp16 --nheads <num_attention_heads> --saveEngine /path/to/xxx.plan
    ```
3. In TensorRT/demo/BERT/build, you have libbert_plugins.so and libcommon.so