# -*- coding: utf-8 -*-
import caffe
from caffe.proto import caffe_pb2


def array_to_mtcnndatum(img_data, label, roi, pts=None):
    """Convert data to mtcnnDatum
    """
    mtcnn_datum = caffe_pb2.MTCNNDatum()
    datum = caffe_pb2.Datum()
    datum = caffe.io.array_to_datum(img_data.transpose((2, 0, 1)), label)  # rgb
    # datum = caffe.io.array_to_datum(img_data, label)  #bgr

    # copy data
    mtcnn_datum.datum.data = datum.data
    mtcnn_datum.datum.label = datum.label

    mtcnn_datum.datum.channels = datum.channels
    mtcnn_datum.datum.height = datum.height
    mtcnn_datum.datum.width = datum.width

    # set roi data
    mtcnn_datum.rois.xmin = float(roi[0])
    mtcnn_datum.rois.ymin = float(roi[1])
    mtcnn_datum.rois.xmax = float(roi[2])
    mtcnn_datum.rois.ymax = float(roi[3])

    if pts is not None:
        mtcnn_datum.pts = pts

    return mtcnn_datum

def array_to_ldmarkdatum(img_data, label, ldmark):
    """Convert data to mtcnnDatum 
    """
    ldmark_datum = caffe_pb2.LdmarkDatum()
    datum = caffe_pb2.Datum()
    #datum= caffe.io.array_to_datum(img_data.transpose((2,0,1)), label)  # rgb
    #datum = caffe.io.array_to_datum(img_data, label)  #bgr

    #copy data
    ldmark_datum.datum.data = datum.data
    ldmark_datum.datum.label = datum.label
    
    ldmark_datum.datum.channels = datum.channels
    ldmark_datum.datum.height = datum.height
    ldmark_datum.datum.width = datum.width

    if ldmark is not None:
        ldmark_datum.pts = ldmark
    
    return ldmark_datum
    
