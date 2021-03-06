import os, cv2, numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import struct
from shutil import copyfile

# video_path = '/media/data/datasets/Kinect2017-10/Datasets/'
video_path = '/Users/trungnd/pfiev/yolo-trainer/video/'
out_path = '/Users/trungnd/pfiev/yolo-trainer/output/'
gt_root = '/Users/trungnd/pfiev/yolo-trainer/data/'

def read_depth(file_reader):
    try:
        byte = file_reader.read(4)
        val = struct.unpack('=L', byte)
        # print 'img size: '+str(val[0])
        img_data = np.fromstring(file_reader.read(val[0]), dtype=np.uint8)
        depth = cv2.imdecode(img_data, cv2.IMREAD_ANYDEPTH)
        # depth=cv2.resize(depth,(640/3,480/3))
        # depth=np.right_shift(depth,8)
        depth = np.left_shift(depth, 5)

        depth = np.asarray(depth, np.uint8)
        ret, mask = cv2.threshold(depth, 1, 255, cv2.THRESH_BINARY)
        # print mask.shape

        # depth=255-depth
        # ret,depth = cv2.threshold(depth,254,255,cv2.THRESH_TOZERO_INV)

        # depth_vis=cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        # depth=cv2

        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contours) > 0):
            cnt = contours[0]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask
    except Exception as e:
        print
        'Error read depth: ' + str(e)
        return None


gt_files = []
for _, _, files in os.walk(gt_root):
    gt_files = files
    break

for path in gt_files:
    print path
    print(gt_root+path)
    f = open(gt_root + path, 'rt')
    lines = f.read().strip().split('\n')
    file_name = path.split('.')[0]
    print file_name

    video_dir = out_path + file_name
    print video_dir

    # if not os.path.exists(video_dir):
    # 	os.makedirs(video_dir)
    # else:
    # 	continue

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(out_path + file_name + '_mask.mp4', fourcc, 20.0, (640, 480))
    print video_path + 'Kinect_3/depth.bin'
    f_d = open(gt_root + path, 'rb')

    frame_num = 0
    # copyfile(video_path+file_name+'/Kinect_3/color.avi', './data/'+file_name+'.avi')
    while True:
        img = read_depth(f_d)
        # print img.shape
        out.write(img)
        # cv2.imwrite(video_dir+'/frame_'+str(frame_num)+'.png',img)
        # plt.imshow(img)
        # plt.show()
        frame_num += 1
        print("Frame: ", frame_num)
        if frame_num > 1000:
            break
        if img is None:
            break

    # break
    # pass

    break

# print lines
# break