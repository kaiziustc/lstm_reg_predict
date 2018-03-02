#!/usr/bin/env sh
/disk/HAnFeng/caffe-LSTM/build/tools/caffe train \
    --solver=lstm_reg_predict_solver.prototxt  > lstm_reg_predict9_lr01_step10000_weight001_3_1.log 2>&1 #--snapshot=lstm__iter_31000.solverstate

