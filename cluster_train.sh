#!/bin/bash
path=$PWD
${AFO_TF_HOME}/bin/tensorflow-submit.sh -conf ${path}/textCNN_estimator.xml -files ${path}/textCNN_estimator_args.py,viewfs://hadoop-meituan/user/hadoop-consec-algo/yanjiamei/data/embed_weights/embedding.zip
