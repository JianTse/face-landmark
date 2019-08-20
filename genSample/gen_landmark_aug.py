# coding: utf-8
import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
import numpy.random as npr
from landmark_augment import LandmarkAugment
from landmark_helper import LandmarkHelper

font = cv2.FONT_HERSHEY_SIMPLEX
__landmark_augment = LandmarkAugment()
__landmark_helper = LandmarkHelper()
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
'''
def GenerateData(ftxt, out_txt, out_dir,img_size,argument=False, landmark_num=5):
    #需要修改
    #1：将所有原始数据全部都flip
    #2：数据集由原来的+flip共同组成
    #3：将新的数据集上做数据增强，包括随机平移、缩放、旋转
    #4：在做数据增强的时候，要保证所有的ldmark点都在样本图像内部，
    
    size = img_size
    image_id = 0
    continue_cnt = 0
    #f = open(join(OUTPUT,"landmark_%s_aug.txt" %(size)),'w')
    f = open(out_txt, "w")
    data = getDataFromTxt(ftxt, landmark_num=landmark_num)
    idx = 0
    #image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in data:
        #print imgPath
        F_imgs = []
        F_landmarks = []        
        img = cv2.imread(imgPath)
        assert(img is not None)
        img_h,img_w,img_c = img.shape
        gt_box = np.array([bbox.left,bbox.top,bbox.right,bbox.bottom])
        f_face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
        f_face = cv2.resize(f_face,(size,size))
        landmark = np.zeros((landmark_num, 2))
        #normalize
        for index, one in enumerate(landmarkGt):
            rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            landmark[index] = rv
        
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(landmark_num*2))
        landmark = np.zeros((landmark_num, 2))        
        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print idx, "images done"
            x1, y1, x2, y2 = gt_box
            #gt's width
            gt_w = x2 - x1 + 1
            #gt's height
            gt_h = y2 - y1 + 1        
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            #random shift
            for i in range(50):  #10
                bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = max(x1+gt_w/2-bbox_size/2+delta_x,0)
                ny1 = max(y1+gt_h/2-bbox_size/2+delta_y,0)
                
                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue_cnt = continue_cnt+1
                    print("continue_cnt:{}, bbox_size:{}, delta_x:{}, delat_y:{}, nx1:{}, ny1:{}, nx2:{}, ny2:{}, img_w:{} img_h：{}".format(continue_cnt, bbox_size, delta_x, delta_y, nx1, ny1, nx2, ny2, img_w, img_h))
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]
                resized_im = cv2.resize(cropped_im, (size, size))
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    #normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0]-nx1)/bbox_size, (one[1]-ny1)/bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(landmark_num*2))
                    landmark = np.zeros((landmark_num, 2))
                    landmark_ = F_landmarks[-1].reshape(-1,2)
                    bbox = BBox([nx1,ny1,nx2,ny2])                    

                    #mirror                    
                    if random.choice([0,1,2]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        #c*h*w
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2))                    
                    #rotate
                    if random.choice([0,1,2]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 5)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2))                
                    
                    #inverse clockwise rotation
                    if random.choice([0,1,2]) > 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -5)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2)) 

                    #big angle
                    #rotate
                    if random.choice([0,1,2]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 10)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2))                
                    
                    #inverse clockwise rotation
                    if random.choice([0,1,2]) > 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -10)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2)) 

                    #rotate
                    if random.choice([0,1,2]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 20)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2))                
                    
                    #inverse clockwise rotation
                    if random.choice([0,1,2]) > 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -20)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2)) 
                    #rotate
                    if random.choice([0,1,2]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), 30)#逆时针旋转
                        #landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        #flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2))                
                    
                    #inverse clockwise rotation
                    if random.choice([0,1,2]) > 0: 
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -30)#顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(landmark_num*2))
                
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(landmark_num*2)) 
                    


            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
            #print F_imgs.shape
            #print F_landmarks.shape
            for i in range(len(F_imgs)):
                print image_id
                cv2.imwrite(join(out_dir,"%d.jpg" %(image_id)), F_imgs[i])
                landmarks = map(str,list(F_landmarks[i]))
                f.write(join(out_dir,"%d.jpg" %(image_id)) + " " +" ".join(landmarks)+"\n")
                image_id = image_id + 1
     
    f.close()
    print("continue_cnt:{}".format(continue_cnt))
    return F_imgs,F_landmarks
'''
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
    data = map(float, param[0:196])
    ret = {}
    ret['imgDir'] = imgDir
    ret['imgFn'] = param[196]
    path, ret['ldmark'] = __landmark_helper.parse(imgDir, line, 98)
    return ret


def showSample(image_new, landmarks_new):
    landmarks = landmarks_new.reshape([-1, 2])
    for l in landmarks:
        ii = tuple(l * (112, 112))
        cv2.circle(image_new, (int(ii[0]), int(ii[1])), 2, (0, 255, 0), -1)
    cv2.imshow('sample', image_new)
    cv2.waitKey(1)

def saveSample(image_new, landmarks_new, sampleDir, srcImgIdx, augIdx, saveFn):
    imgFn = sampleDir + '/%d_%d.jpg' % (srcImgIdx, augIdx)
    retStr = infoToStr(imgFn, landmarks_new)
    saveFn.write(retStr + '\n')
    cv2.imwrite(imgFn, image_new)

def genAugSample(image, ldmarks,saveF, sampleDir, srcImgIdx):
    for i in range(2):
        image_new, landmarks_new = __landmark_augment.augment(image, ldmarks, 112, 20, (1.1, 1.3))
        saveSample(image_new, landmarks_new, sampleDir, srcImgIdx, i+1, saveF)
        showSample(image_new, landmarks_new)

def GenerateData(srcImgDir, srcList):
    sampleDir = 'E:/work/data/landmark/samples/98/testSample_98'
    saveF = open("E:/work/data/landmark/samples/98/test_98_list.txt", "w")
    for idx in range(len(srcList)):
        #info = readTrainInfo(srcImgDir, srcList[idx])
        info = readTestInfo(srcImgDir, srcList[idx])
        image = cv2.imread(info['imgDir'] + info['imgFn'])
        image_new, landmarks_new = __landmark_augment.augment(image, info['ldmark'], 112, 20, (1.1, 1.3))
        saveSample(image_new, landmarks_new, sampleDir, idx, 0, saveF)
        showSample(image_new, landmarks_new)
        genAugSample(image, info['ldmark'], saveF, sampleDir, idx)

    saveF.close()

def readAllGT():
    srcImgDir = 'E:/work/data/landmark/WFLW/WFLW_images/WFLW_images/'
    srcFn = 'list_98pt_test.txt'
    srcList = load_file_list(srcFn)
    '''
    for idx in range(len(srcList)):
        info = readTestInfo(srcImgDir, srcList[idx])
        img = cv2.imread(info['imgDir'] + info['imgFn'])
        drawPts(img, info, 'img')
        cv2.waitKey(0)
    '''
    GenerateData(srcImgDir, srcList)

if __name__ == '__main__':
    readAllGT()
