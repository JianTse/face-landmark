#coding:utf-8
import os
import random
import sys
import time
import h5py
import cv2
import numpy as np
from multiprocessing import Pool
import linecache

def mean_per_img(img):
    b_m = np.mean(img[:,:,0])
    g_m = np.mean(img[:,:,1])
    r_m = np.mean(img[:,:,2])
    
    return b_m, g_m, r_m

def mean_imgs(img_dir, height=112, width=112):
    b_mean_list = []
    g_mean_list = []
    r_mean_list = []
    for f in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, f))
        img = cv2.resize(img, (width, height))
        img = np.array(img).astype(np.float32)
        b_m, g_m, r_m = mean_per_img(img)
        b_mean_list.append(b_m)
        g_mean_list.append(g_m)
        r_mean_list.append(r_m)
    return np.mean(b_mean_list), np.mean(g_mean_list), np.mean(r_mean_list)

def std_per_img(img, b_m, g_m, r_m):

    b_var = np.mean( (img[:,:,0] - b_m)*(img[:,:,0] - b_m) )
    g_var = np.mean( (img[:,:,1] - g_m)*(img[:,:,1] - g_m) )
    r_var = np.mean( (img[:,:,2] - r_m)*(img[:,:,2] - r_m) )
    
    return np.sqrt(b_var), np.sqrt(g_var), np.sqrt(r_var)

def std_imgs(img_dir, b_m, g_m, r_m, height=112, width=112):
    b_std_list = []
    g_std_list = []
    r_std_list = []
    for f in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, f))
        img = cv2.resize(img, (width, height))
        img = np.array(img).astype(np.float32)
        b_std, g_std, r_std = std_per_img(img, b_m, g_m, r_m)
        
        b_std_list.append(b_std)
        g_std_list.append(g_std)
        r_std_list.append(r_std)
    return np.mean(b_std_list), np.mean(g_std_list), np.mean(r_std_list)


def norm(img):
    bm, gm, rm = (135.70923, 142.39746, 169.96283)
    bstd, gstd, rstd = (63.605362, 64.640724, 75.43583)
    img = np.asarray(img).astype(np.float32)
    img[:,:,0] = (img[:,:,0]-bm)/bstd
    img[:,:,1] = (img[:,:,1]-gm)/gstd
    img[:,:,2] = (img[:,:,2]-rm)/rstd

    return img


def get_dataset(imglists_path, landmark_num=5):
    imagelist = open(imglists_path, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        
        #data_example['filename'] = os.path.join("../../data/preproc",info[0])
        data_example['filename'] = info[0]
        landmark = []

        assert len(info)==1+landmark_num*2 and landmark_num!=2, "Length of info {} or landmark_num {} is error".format(len(info), landmark_num)
        
        for i in range(1, len(info)):
            landmark.append(float(info[i]))

        data_example['landmark'] = landmark
        dataset.append(data_example)

    imagelist.close()
    return dataset


def batch_generator(dataset, batch_size):
    num = len(dataset)
    for i in range(0, num, batch_size):
        pad = i + batch_size - num
        pad = pad if pad >0 else 0
        batch = dataset[i:i+batch_size] + random.sample(dataset[0:i], pad)
        yield batch

def batch_gen(path, batch_size, shuffle=True, landmark_num=127):
    num = 0
    f = open(path, "r")
    while True:
        line = f.readline()
        if line == '':
            break
        num = num + 1
    f.close()
    line_indexes = list(range(1,num+1))
    if shuffle:
        random.shuffle(line_indexes)
    for i in range(0, num, batch_size):
        pad = i + batch_size - num
        pad = pad if pad >0 else 0
        #batch = dataset[i:i+batch_size] + random.sample(dataset[0:i], pad)
        index_batch = list(range(i,min(num,i+batch_size))) + random.sample(list(range(0,num)), pad)

        batch = []
        for index in index_batch:
            print("index", index)
            # print("line_indexes[index]",line_indexes[index])
            line = linecache.getline(path, line_indexes[index])
            info = line.strip().split(' ')
            data_example = dict()
            data_example['filename'] = info[0]
            landmark = []
            assert len(info)==1+landmark_num*2 and landmark_num!=2, "Length of info {} or landmark_num {} is error".format(len(info), landmark_num)
            for i in range(1, len(info)):
                landmark.append(float(info[i]))
            data_example['landmark'] = landmark

            batch.append(data_example)

        yield batch

def get_hdf5(imglists_path, output_txt, hdf5_dir, img_size, batch_size, shuffling=False, landmark_num=5):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    # GET Dataset, and shuffling.
    dataset = get_dataset(imglists_path, landmark_num=landmark_num)
    # filenames = dataset['filename']
    if shuffling:
        prefix = "shuffle_"
        random.shuffle(dataset)
    else:
        prefix = ""

    bg = batch_generator(dataset, batch_size)

    imgDir = "E:/work/data/landmark/samples/98/"
    txt = open(output_txt, 'w')
    
    img_array = np.zeros((batch_size, 3, img_size, img_size)).astype(np.float32)
    label_array = np.zeros((batch_size, landmark_num*2)).astype(np.float32)
    for i,batch in enumerate(bg):
        for j,item in enumerate(batch):
            imgFn = imgDir + item['filename']
            img = cv2.imread(imgFn)
            if img is None:
                print("Warning:cv2.imread {} is None".format(item['filename']))
                img = np.zeros((3, img_size, img_size)).astype(np.float32)
                label = np.zeros((landmark_num*2)).astype(np.float32)
            else:
                h,w,c = img.shape
                if w!=img_size or h!=img_size:
                    img = cv2.resize(img, (img_size, img_size))

                img = (np.asarray(img).astype(np.float32) - 127.5)/128.0
                img = np.transpose(img, (2,0,1))    #convert (height, width, 3) to (3, height, width)
                label = np.array(item['landmark'])
            img_array[j, :, :, :] = img
            label_array[j, ...] =label

        filename = prefix + "batch_" + str(i) + ".hdf5"
        with h5py.File(os.path.join(hdf5_dir,filename), 'w') as f:
            f.create_dataset('data', data=np.asarray(img_array).astype(np.float32))            
            f.create_dataset('label', data=np.asarray(label_array).astype(np.float32))
       
        txt.writelines(filename+'\n')
        print(filename)

    txt.close()


def gen_labelmap(label, size):
    labelmap = np.zeros((size, size)).astype(np.float32)
    label = label.reshape((-1,2))
    for p in label:
        x = int(p[0]*(size-1))
        y = int(p[1]*(size-1))
        
        if x>=0 and x<size and y>=0 and y<size:
            labelmap[y][x] = 1
    
    return labelmap

def gen_batch(batch, proc_fn=norm):
    landmark_num = 127
    batch_size = len(batch)
    hdf5_name = os.path.splitext(os.path.split(batch[0]['filename'])[-1])[0] + ".hdf5"
    print("hdf5_name:{}".format(hdf5_name))

    img_array = np.zeros((batch_size, 3, img_size, img_size)).astype(np.float32)
    label_array = np.zeros((batch_size, landmark_num*2)).astype(np.float32)

    for j,item in enumerate(batch):
        
        path = os.path.join("/data/proj/FaceLandmark/fast-facial-landmark-detection/data/preproc", item['filename'])
        img = cv2.imread(path)
            
        if img is None:
            print("Warning:cv2.imread {} is None".format(item['filename']))
            img = np.zeros((3, img_size, img_size)).astype(np.float32)
            label = np.zeros((landmark_num*2)).astype(np.float32)
        else:
            w,h,c = img.shape
            if w!=img_size or h!=img_size:
                img = cv2.resize(img, (img_size, img_size))

            # img = (np.asarray(img).astype(np.float32) - 127.5)/128.0
            if proc_fn is None:
                img = np.asarray(img).astype(np.float32)
            else:
                img = proc_fn(img)
            img = np.transpose(img, (2,0,1))    #convert (height, width, 3) to (3, height, width)
            label = np.array(item['landmark'])
            # labelmap = gen_labelmap(label, img_size)

        img_array[j, ...] = img
        label_array[j, ...] =label

    # with h5py.File(os.path.join(hdf5_dir,hdf5_name), 'w') as f:
    f = h5py.File(os.path.join(hdf5_dir,hdf5_name), 'w')
    f.create_dataset('data', data=np.asarray(img_array).astype(np.float32))            
    f.create_dataset('label', data=np.asarray(label_array).astype(np.float32))
    f.close()

def get_batch_heatmap_points(batch):

    batch_size = len(batch)
    landmarks_num = 127
    hdf5_name = os.path.splitext(os.path.split(batch[0]['filename'])[-1])[0] + ".hdf5"
    print("hdf5_name:{}".format(hdf5_name))

    img_array = np.zeros((batch_size, 3, img_size, img_size)).astype(np.float32)
    heatmap_array = np.zeros((batch_size, img_size, img_size)).astype(np.float32)
    points_array = np.zeros((batch_size, landmarks_num*2)).astype(np.float32)

    for j,item in enumerate(batch):
        img = cv2.imread(item['filename'])
            
        if img is None:
            print("Warning:cv2.imread {} is None".format(item['filename']))
            img = np.zeros((3, img_size, img_size)).astype(np.float32)
            label = np.zeros((img_size, img_size)).astype(np.float32)
        else:
            w,h,c = img.shape
            if w!=img_size or h!=img_size:
                img = cv2.resize(img, (img_size, img_size))

            img = (np.asarray(img).astype(np.float32) - 127.5)/128.0
            img = np.transpose(img, (2,0,1))    #convert (height, width, 3) to (3, height, width)
            label = np.array(item['landmark'])

        img_array[j, ...] = img
        points_array[j, ...] = label

        heatmap = gen_labelmap(label, img_size)
        heatmap_array[j, ...] = heatmap

    with h5py.File(os.path.join(hdf5_dir, hdf5_name), 'w') as f:
        f.create_dataset('data', data=np.asarray(img_array).astype(np.float32))            
        f.create_dataset('heatmap', data=np.asarray(heatmap_array).astype(np.float32))
        f.create_dataset('points', data=np.asarray(points_array).astype(np.float32))
    
    print("one batch done")


def gen_end2end(imglists, batch_size, shuffle=True, proc_fn=None):
    '''
    dataset = get_dataset(imglists, landmark_num=127)
    if shuffle:
        random.shuffle(dataset)

    bg = batch_generator(dataset, batch_size)

    
    for i,batch in enumerate(bg):
        #gen_batch(batch)
        get_batch_heatmap_points(batch)
        print("batch {} done.".format(i))
    '''

    bg = batch_gen(imglists, batch_size, shuffle=shuffle, landmark_num=127)
    
    pool = Pool(12)
    pool.map(gen_batch, bg)
    pool.close()
    pool.join()
    '''
    for i,batch in enumerate(bg):
        gen_batch(batch, proc_fn)
        # get_batch_heatmap_points(batch)
        print("batch {} done.".format(i))
    '''
    print("All done.")    


if __name__ == '__main__':

    #imglists_path = "data/landmark_48_aug.txt"
    #output_path = "data/tfdata/landmark_data.tfrecord"

    img_size = 64
    #batch_size = 384
    # img_size = 80
    batch_size = 8
    data_dir = "E:/work/data/landmark/samples/98/test_hdf5"
    #imglists_path = os.path.join(data_dir, str(img_size), "landmark_aug.txt")
    #imglists_path = "E:/work/data/landmark/samples/train_98_list.txt"
    imglists_path = "E:/work/data/landmark/samples/98/test_98_list.txt"

    #out_dir = "hdf5_2"
    #out_dir = "hdf5-90"
    #out_dir = "hdf5-end2end112"
    # out_dir = "hdf5-headmap-points"
    out_dir = "hdf5-norm"
    output_txt =  os.path.join(data_dir, str(img_size),  out_dir+".txt")
    hdf5_dir = os.path.join(data_dir, str(img_size),  out_dir)

    #if not os.path.exists(os.path.split(output_txt)[0]):
    #    os.makedirs(os.path.split(output_txt)[0])

    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    
    
    get_hdf5(imglists_path, output_txt, hdf5_dir, img_size, batch_size, shuffling=True, landmark_num=98)

    #gen_end2end(imglists_path, batch_size, shuffle=True, proc_fn=norm)
