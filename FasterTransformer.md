FasterTransformer github: https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer

1. Using code in v2 (same as v1 for encoder)
2. Follow the Quick Start Guide to Build the FasterTransformer, this step will generate "./lib/libtf_fastertransformer.so"
3. Follow the Quick Start Guide to generate the gemm_config.in by calling " ./bin/encoder_gemm 1 128 12 64 <is_use_fp16>"
4. copy profile_bert_squad_inference.py to sample/tensorflow_bert/, it's modified based on profile_bert_inference.py
5. Modify profile_util.py's run_profile funtion to export model.
    ```python
    def run_profile(graph_fn, jit_xla, num_iter, profiler=None, init_checkpoint=None, check_result=True, dryrun_iter=1, export_path=None):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = jit_xla

        fetches, features = graph_fn()

        with tf.Session(config=config) as sess:
            # init
            if init_checkpoint is None:
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.Saver()
                saver.restore(sess, init_checkpoint)

            # dry run
            for _ in range(dryrun_iter):
                sess.run(fetches)

            # you can also use simple save.
            input_ids = tf.saved_model.utils.build_tensor_info(features["input_ids"])
            input_mask = tf.saved_model.utils.build_tensor_info(features["input_mask"])
            segment_ids = tf.saved_model.utils.build_tensor_info(features["segment_ids"])
            prediction = tf.saved_model.utils.build_tensor_info(fetches)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_ids': input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids},
                    outputs={'prediction': prediction},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)) #/predict/prediction
            # export model
            if export_path is not None:
                print('Exporting trained model to', export_path)
                if os.path.isdir(export_path) and not os.path.islink(export_path):
                    shutil.rmtree(export_path)
                os.makedirs(export_path)

                builder = tf.saved_model.builder.SavedModelBuilder(export_path)
                builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature
                        },
                    strip_default_attrs=True,
                    clear_devices=True)
                builder.save()

            res = []
            if profiler is None:
                start_time = time.time()
                if check_result:
                    for _ in range(num_iter):
                        res.append(sess.run(fetches))
                else:
                    for _ in range(num_iter):
                        sess.run(fetches)
                end_time = time.time()
                time_avg = (end_time - start_time)/num_iter

            else:
                if check_result:
                    for _ in range(num_iter):
                        with profiler.prof_run():
                            res.append(sess.run(fetches, options=profiler.run_options, run_metadata=profiler.run_metadata))
                else:
                    for _ in range(num_iter):
                        with profiler.prof_run():
                            sess.run(fetches, options=profiler.run_options, run_metadata=profiler.run_metadata)
                time_avg = profiler.time_avg

        return time_avg, res
    ```
6. Download glue data: https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks  
    Download bert checkpoint here: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
7. export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
8. export GLUE_DIR=/path/to/glue
9. copy https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/run_classifier.py to sample/tensorflow_bert/
10. cd sample/tensorflow_bert/ and run sample with: 
   ```bash
   python profile_bert_squad_inference.py   --task_name=MRPC   --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --predict_batch_size=1   --max_seq_length=128   --output_dir=mrpc_output --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --tf_profile=false  --profiling_output_file=time_elapsed --xla=false --floatx=float32
   ```
11. find exported model in /exported_ft_xx
12. for FP16 models, follow sample.md in https://github.com/NVIDIA/DeepLearningExamples/blob/master/FasterTransformer/v2/sample/tensorflow_bert/sample.md, you have to run
    ```bash
    python ckpt_type_convert.py --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --fp16_checkpoint=$BERT_BASE_DIR/bert_model_fp16.ckpt
    ```
    Remember to modify --floatx and --init_checkpoint in step 9 to generate correct model files in fp16
