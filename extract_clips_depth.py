import os, cv2, numpy as np, matplotlib.pyplot as plt
import struct
from shutil import copyfile

video_path = '/media/data/datasets/Kinect2017-10/Datasets/'
out_path = './data/'
gt_root = '/media/data/datasets/Kinect_vis_annotate/v5/Kinect3/'


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

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # print mask.shape
        return mask
    except Exception as e:
        print
        'Error: ' + str(e)
        return None


gt_files = []
for _, _, files in os.walk(gt_root):
    gt_files = files
    break

for path in gt_files:
    # print path
    f = open(gt_root + path, 'rt')
    lines = f.read().strip().split('\n')
    file_name = path.split('.')[0]

    video_dir = out_path + file_name
    # print video_dir
    # if not os.path.exists(video_dir):
    # 	os.makedirs(video_dir)
    # else:
    # 	continue

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('./data/' + file_name + '_mask.mp4', fourcc, 20.0, (640, 480))

    f_d = open(video_path + file_name + '/Kinect_3/depth.bin', 'rb')

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
        if img is None:
            break

    # break
    # pass

    break

# print lines
# break

