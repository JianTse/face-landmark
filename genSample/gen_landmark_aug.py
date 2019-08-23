# coding: utf-8
import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
import numpy.random as npr
from gen_hdf5 import genAllHDF5
from landmark_augment import LandmarkAugment
from landmark_helper import LandmarkHelper
from euler_angles import PnpHeadPoseEstimator

font = cv2.FONT_HERSHEY_SIMPLEX
__landmark_augment = LandmarkAugment()
__landmark_helper = LandmarkHelper()

def estimatorPose(image, ldmark127):
    __estimator = PnpHeadPoseEstimator(cam_w=image.shape[1], cam_h=image.shape[0])
    points_to_return = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    landmarks = np.zeros((len(points_to_return), 2), dtype=np.float32)
    counter = 0
    for point in points_to_return:
        landmarks[counter] = [ldmark127[point][0], ldmark127[point][1]]
        counter += 1
    pitch_yaw_roll = __estimator.return_pitch_yaw_roll(landmarks)
    pitch, yaw, roll = map(lambda x: x[0], pitch_yaw_roll)
    return [pitch, yaw, roll]

def load_file_list(anno_fn):
    list = []
    with open(anno_fn, 'r') as f_anno:
        while True:
            line = f_anno.readline()
            if not line:
                break
            file_name = line.strip()
            list.append(file_name)
    return list

def drawPts(img, gt_info, name):
    for idx in range(len(gt_info['ldmark'])):
        x = gt_info['ldmark'][idx][0]
        y = gt_info['ldmark'][idx][1]
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        #cv2.rectangle(img, (gt_info['rect'][0], gt_info['rect'][1]), (gt_info['rect'][2], gt_info['rect'][3]),(0, 0, 255), 2)
        cv2.putText(img, '%d' % (idx), (int(x), int(y)), font, 0.3, (0, 0, 255), 1)
    cv2.imshow(name, img)


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
     # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter*1.0 / (box_area + area - inter)
    return ovr

def flip_landmark(imgWidth, landmark):
    """
        flip landmark
    """

    def swap_points_sym(a, b):
        tmp = a.copy()
        a = b[::-1]
        b = tmp[::-1]
        return a, b

    def swap_points(a, b):
        tmp = a.copy()
        a = b
        b = tmp
        return a, b

    # mirror
    landmark_ = np.asarray([(imgWidth - x, y) for (x, y) in landmark])

    # 5 landmakr points
    if len(landmark) == 5:
        landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
        landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth

    # 127 landmark points
    elif len(landmark) == 127:
        landmark_[0:8], landmark_[9:17] = swap_points_sym(landmark_[0:8], landmark_[9:17])
        landmark_[17:22], landmark_[22:27] = swap_points_sym(landmark_[17:22], landmark_[22:27])
        landmark_[31:33], landmark_[34:36] = swap_points_sym(landmark_[31:33], landmark_[34:36])
        # landmark_[36:42], landmark_[42, 48] = swap_points_sym(landmark_[36:42], landmark_[42, 48])
        landmark_[36:40], landmark_[42:46] = swap_points_sym(landmark_[36:40], landmark_[42:46])
        landmark_[40:42], landmark_[46:48] = swap_points_sym(landmark_[40:42], landmark_[46:48])

        landmark_[48:51], landmark_[52:55] = swap_points_sym(landmark_[48:51], landmark_[52:55])
        landmark_[55:57], landmark_[58:60] = swap_points_sym(landmark_[55:57], landmark_[58:60])
        landmark_[60:62], landmark_[63:65] = swap_points_sym(landmark_[60:62], landmark_[63:65])
        landmark_[65], landmark_[67] = swap_points(landmark_[65], landmark_[67])
        landmark_[68:71], landmark_[71:74] = swap_points_sym(landmark_[68:71], landmark_[71:74])
        landmark_[74:79], landmark_[84:89] = swap_points_sym(landmark_[74:79], landmark_[84:89])
        landmark_[79:84], landmark_[89:94] = swap_points_sym(landmark_[79:84], landmark_[89:94])
        # landmark_[94:99], landmark_[99:104] = swap_points_sym(landmark_[94:99], landmark_[99:104])
        landmark_[94], landmark_[101] = swap_points(landmark_[94], landmark_[101])
        landmark_[95], landmark_[100] = swap_points(landmark_[95], landmark_[100])
        landmark_[96], landmark_[99] = swap_points(landmark_[96], landmark_[99])
        landmark_[97], landmark_[102] = swap_points(landmark_[97], landmark_[102])
        landmark_[98], landmark_[103] = swap_points(landmark_[98], landmark_[103])

        landmark_[104:109], landmark_[109:114] = swap_points_sym(landmark_[104:109], landmark_[109:114])
        landmark_[114:118], landmark_[118:122] = swap_points(landmark_[114:118], landmark_[118:122])
        landmark_[122], landmark_[123] = swap_points(landmark_[122], landmark_[123])
        landmark_[125], landmark_[126] = swap_points(landmark_[125], landmark_[126])

    else:
        print("Warning: Lenth of landmark is invalid.")

    return landmark_

def infoToStr(imgFn, ldmark):
    retStr = imgFn
    for idx in range(len(ldmark)):
        retStr += ' ' + str(ldmark[idx])
    return retStr

def readTrainInfo(imgDir, line):
    param = line.split()
    data = map(float, param[0:196])
    ret = {}
    ret['imgDir'] = imgDir
    ret['imgFn'] = param[206]
    ret['rect'] = [int(param[196]),int(param[197]), int(param[198]), int(param[199])]
    ret['att'] = [int(param[200]),int(param[201]), int(param[202]), int(param[203]),int(param[204]),int(param[205])]
    path, ret['ldmark'] = __landmark_helper.parse(imgDir, line, 98)
    return ret

def readTestInfo(imgDir, line):
    param = line.split()
    ret = {}
    ret['imgDir'] = imgDir
    ret['imgFn'] = param[0]
    ret['eva'] = [float(param[255]), float(param[256]), float(param[257])]
    path, ret['ldmark'] = __landmark_helper.parse(imgDir, line, 127)
    return ret

def flip(img, gt_info):
    flip_img = cv2.flip(img, 1)
    eva = gt_info['eva']
    ret = {}
    ret['imgDir'] = gt_info['imgDir']
    ret['imgFn'] = gt_info['imgFn']
    ret['eva'] = [eva[0], -1*eva[1], -1*eva[2]]
    ret['ldmark'] = flip_landmark(img.shape[1], gt_info['ldmark'])
    return flip_img, ret

def showSample(image_new, landmarks_new):
    landmarks = landmarks_new.reshape([-1, 2])
    for l in landmarks:
        ii = tuple(l * (112, 112))
        cv2.circle(image_new, (int(ii[0]), int(ii[1])), 2, (0, 255, 0), -1)
    cv2.imshow('sample', image_new)
    cv2.waitKey(0)

def saveSample(image_new, landmarks_new, sampleDir, srcImgIdx, augIdx, saveFn, type):
    if type == 0:
        imgFn = '%d_%d.jpg' % (srcImgIdx, augIdx)
    else:
        imgFn = '%d_%d_flip.jpg' % (srcImgIdx, augIdx)
    retStr = infoToStr(imgFn, landmarks_new)
    saveFn.write(retStr + '\n')
    saveImgFn = sampleDir + '/' + imgFn
    cv2.imwrite(saveImgFn, image_new)

def genAugSample(image, ldmarks,saveF, sampleDir, srcImgIdx, type):
    for i in range(14):
        minAngle = (i - 7) * 5
        maxAngle = minAngle + 5
        print ('%d: [%d:%d]'%(i, minAngle, maxAngle))
        angleRange = [minAngle,maxAngle]
        image_new, landmarks_new = __landmark_augment.augment(image, ldmarks, 112, angleRange, (1.1, 1.3))
        saveSample(image_new, landmarks_new, sampleDir, srcImgIdx, i, saveF,type)
        #showSample(image_new, landmarks_new)

def genAugSample_new(image, ldmarks, eva, saveF, sampleDir, srcImgIdx, type):
    #pitch, yaw, roll = estimatorPose(image, ldmarks)
    start_angle = -35 + eva[2] #方向反了
    end_angle = 35 + eva[2] #方向反了
    for i in range(16):
        minAngle = start_angle + i * 5
        maxAngle = minAngle + 5
        if minAngle >= start_angle and maxAngle <= end_angle:
            #print ('good: %d: [%d:%d]'%(i, minAngle, maxAngle))
            angleRange = [minAngle,maxAngle]
            image_new, landmarks_new = __landmark_augment.augment(image, ldmarks, 112, angleRange, (1.1, 1.3))
            saveSample(image_new, landmarks_new, sampleDir, srcImgIdx, i, saveF,type)
            #showSample(image_new, landmarks_new)
        #else:
        #    print ('bad: %d: [%d:%d]' % (i, minAngle, maxAngle))

def GenerateData(srcImgDir, srcList, dstSampleDir, dstSampleListFn):
    if not os.path.exists(dstSampleDir):
        os.makedirs(dstSampleDir)
    saveF = open(dstSampleListFn, "w")
    for idx in range(len(srcList)):
        print ('%d, %d'%(len(srcList),idx))
        info = readTestInfo(srcImgDir, srcList[idx])
        image = cv2.imread(info['imgDir'] + info['imgFn'])
        #genAugSample(image, info['ldmark'], saveF, dstSampleDir, idx, 0)
        genAugSample_new(image, info['ldmark'], info['eva'], saveF, dstSampleDir, idx, 0)
        flip_img, flip_info = flip(image, info)
        #genAugSample(flip_img, flip_info['ldmark'], saveF, dstSampleDir, idx, 1)
        genAugSample_new(flip_img, flip_info['ldmark'], flip_info['eva'], saveF, dstSampleDir, idx, 1)
    saveF.close()

def readAllGT():
    #rootDir = 'E:/work/data/landmark/beadwallet/'
    rootDir = '/home/sxdz/data/landmark/beadwallet/'
    srcImgDir = rootDir + 'images/'

    srcTrainImgListFn = rootDir + 'samples/train_anno_list.txt'
    dstTrainSampleDir = rootDir + 'samples/trainSample'
    dstTrainSampleListFn = rootDir + 'samples/train_sample_list.txt'
    srcTrainList = load_file_list(srcTrainImgListFn)
    GenerateData(rootDir, srcTrainList, dstTrainSampleDir, dstTrainSampleListFn)
    
    srcTestImgListFn = rootDir + 'samples/test_anno_list.txt'
    dstTestSampleDir = rootDir + 'samples/testSample'
    dstTestSampleListFn = rootDir + 'samples/test_sample_list.txt'
    srcTestList = load_file_list(srcTestImgListFn)
    GenerateData(rootDir, srcTestList, dstTestSampleDir, dstTestSampleListFn)

    srcValImgListFn = rootDir + 'samples/val_anno_list.txt'
    dstValSampleDir = rootDir + 'samples/valSample'
    dstValSampleListFn = rootDir + 'samples/val_sample_list.txt'
    srcValList = load_file_list(srcValImgListFn)
    GenerateData(rootDir, srcValList, dstValSampleDir, dstValSampleListFn)

    #genAllHDF5()

if __name__ == '__main__':
    readAllGT()

    '''
    rootDir = 'E:/work/data/landmark/'
    srcImgDir = rootDir + 'poseSample/'
    srcValImgListFn = rootDir + 'poseSample/anno.txt'
    dstValSampleDir = rootDir + 'poseSample/dstSample'
    dstValSampleListFn = rootDir + 'poseSample/dst_sample_list.txt'
    srcValList = load_file_list(srcValImgListFn)
    GenerateData(rootDir, srcValList, dstValSampleDir, dstValSampleListFn)
    '''