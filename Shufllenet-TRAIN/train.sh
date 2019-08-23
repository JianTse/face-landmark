GLOG_logtostderr=1 ../build/tools/caffe train \
  --solver=solver-112.prototxt \
  --gpu=0  2>&1 | tee log.txt