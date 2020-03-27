1. lock frequency, turn on ECC. nvidia-smi -q -d CLOCK; nvidia-smi -ac 5001,1590 -i x

2. using nvidia-docker and specified image:
   ```bash
    nvidia-docker run --rm -it --name=triton_bert -p8000:8000 -p8001:8001 -v/path/to/model/repo:/models nvcr.io/nvidia/tensorrtserver:19.09-py3
   ```

3. organize your model repository with model files and config.pbtxt
4. export LD_PRELOAD=/path/to/libcommon.so:/path/to/libbert_plugins.so:/path/to/libtf_fastertransformer.so
5. Lauch server:
   ```bash
   trtserver --model-store=/models --log-verbose=1 --strict-model-config=False
   ```
6. In another CLI, run client image:
   ```bash
   docker run --rm -it --net=host nvcr.io/nvidia/tensorrtserver:20.02-py3-clientsdk
   ```
7. In this container, run client(just an example, modify the params according to the help):
   ```bash
   install/bin/perf_client -m bert_trt_fp16 -d -c1 -l2000 -p15000 -b8 -i grpc -u localhost:8001 -t1  --max-threads=64 --input-data=zero
   ```