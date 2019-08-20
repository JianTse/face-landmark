GLOG_logtostderr=1 ../build/tools/caffe train \
  --solver=solver-M3.prototxt \
  --weights=M3.caffemodel \
  --gpu=0  2>&1 | tee log.txt