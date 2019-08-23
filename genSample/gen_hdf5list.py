import os
import random

#  /home/sxdz/data/landmark/98

# /home/sxdz/data/landmark/98/train_hdf5/112/train_hdf5.txt
# /home/sxdz/data/landmark/98/test_hdf5/112/test_hdf5.txt


if __name__ == "__main__":
    txt = "E:/work/data/landmark/samples/98/test_hdf5/64/test_hdf5.txt"

    f = open(txt, "w")

    src_hdf5_dir = "E:/work/data/landmark/samples/98/test_hdf5/64/hdf5-norm/"
    dst_hdf5_dir = "/home/sxdz/data/landmark/98/test_hdf5/64/hdf5-norm/"

    hdf5_list = os.listdir(src_hdf5_dir)

    random.shuffle(hdf5_list)
    for i in hdf5_list:
        #imgFn = phase_fn + i + "\n"
        f.writelines(os.path.join(dst_hdf5_dir, i)+"\n")

    f.close()