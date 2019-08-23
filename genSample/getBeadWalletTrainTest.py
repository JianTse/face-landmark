#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import os
import cv2

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

def getCurType(srcList):
    testIdxList = []
    valIdxList = []
    trainIdxList = []

    totalCount = len(srcList)
    TestValCount = int(totalCount * 0.15)
    step = totalCount / TestValCount
    testValIdxList = []
    for idx in range(totalCount):
        if idx % step == 0:
            testValIdxList.append(idx)
        else:
            trainIdxList.append(idx)
    for idx in range(len(testValIdxList)):
        srcIdx = testValIdxList[idx]
        if idx % 3 == 0:
            valIdxList.append(srcIdx)
        else:
            testIdxList.append(srcIdx)
    return trainIdxList, testIdxList, valIdxList


def  writeSample(imgDir, saveFn, srcList, saveListIdx):
    saveFileFn = open((saveFn), 'w')
    for idx in range(len(saveListIdx)):
        srcIdx = saveListIdx[idx]
        line = srcList[srcIdx]
        #param = line.split(' ')
        print('idx: %d, line: %s' % (idx, line))
        #imgFn = os.path.join(imgDir, param[0])
        #if not os.path.exists(imgFn):
        #    continue
        saveFileFn.write(line + '\n')
    saveFileFn.close()

def filterList():
    rootDir = 'E:/work/data/landmark/beadwallet/samples/'
    srcListFn = rootDir + 'srcPoseAnnoLists_filter.txt'
    trainListFn = rootDir + 'train_anno_list.txt'
    valListFn = rootDir + 'val_anno_list.txt'
    testListFn = rootDir + 'test_anno_list.txt'

    srcList = load_file_list(srcListFn)
    print('total: %d'% (len(srcList)))
    trainIdxList, testIdxList, valIdxList = getCurType(srcList)

    writeSample(rootDir, trainListFn, srcList, trainIdxList)
    writeSample(rootDir, testListFn, srcList, testIdxList)
    writeSample(rootDir, valListFn, srcList, valIdxList)


if __name__ == "__main__":
    filterList()

