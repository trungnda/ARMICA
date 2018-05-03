import os, numpy as np, cv2
import matplotlib
import os
import errno

matplotlib.use('TkAgg')

# root_dir = '/media/data/datasets/Kinect2017-10/Datasets/'
# out_dir = '/home/nguyenductrung/kinect_vis/output/'

root_dir = '/Users/trungnd/pfiev/dataset/'
out_dir = '/Users/trungnd/pfiev/report/'
dataset_dir = '/Users/trungnd/pfiev/dataset/person/'
cluster_img = 10

count_img = 0

# for _, dirs, _ in os.walk(root_dir):
#     break
dirs = ['20171123_Hung_lan1_23-11-2017__11-05-57']
# dirs = ['20171128_Phuong_28-11-2017__10-16-52']
for f_name in dirs:
    # print f_name
    # f_name='20171123_Hung_lan1_23-11-2017__11-05-57'
    vid_dir = root_dir + f_name + '/'
    out_vid_path = out_dir + f_name + '.avi'
    out_photo_prefix = dataset_dir+ f_name+'/'

    print out_photo_prefix
    # if os.path.isfile(out_vid_path):
    #     continue
    print f_name

    import struct


    def read_depth(file_reader):
        try:
            byte = file_reader.read(4)
            val = struct.unpack('=L', byte)
            img_data = np.fromstring(file_reader.read(val[0]), dtype=np.uint8)
            depth = cv2.imdecode(img_data, cv2.IMREAD_ANYDEPTH)

            depth = np.left_shift(depth, 5)

            depth = np.asarray(depth, np.uint8)
            ret, mask = cv2.threshold(depth, 1, 255, cv2.THRESH_BINARY)

            # depth
            # return depth_vis
            return depth
        except Exception as e:
            print 'Error: ' + str(e)
            return None

    color_vids = []
    depth_vids = []
    for i in range(1, 8):

        vid = cv2.VideoCapture(vid_dir + '/Kinect_' + str(i) + '/color.avi')
        color_vids.append(vid)
        print vid_dir + 'Kinect_' + str(i) + '/depth.bin'
        f = open(vid_dir + 'Kinect_' + str(i) + '/depth.bin', "rb")
        depth_vids.append(f)

    print(color_vids)
    print(depth_vids)
    frame_idx = 0
    try:
        while True:
            for kid in range(7):
                if frame_idx % 100 == 0:
                    print 'Frame: ' + str(frame_idx) + ' total img: ' + str(count_img)
                ret, img = color_vids[kid].read()
                depth = read_depth(depth_vids[kid])


                if img is None:
                    break
                if frame_idx % cluster_img == 0:
                    # Process image
                    img_w, img_h, img_chanels = img.shape
                    img_w = float(img_w)
                    img_h = float(img_h)

                    im2, contours, hierarchy = cv2.findContours(depth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
                    maxContourIndex = 0

                    if (len(contours) > 0):
                        for index, contour in enumerate(contours):
                            if (cv2.contourArea(contour) >= cv2.contourArea(contours[maxContourIndex])):
                                maxContourIndex = index
                        x, y, w, h = cv2.boundingRect(contours[maxContourIndex])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

                        count_img += 1
                        cv2.imwrite(out_photo_prefix + 'Kinect_'+ str(kid+1)+ '/'+ str(frame_idx) + '.jpg', img)

                        w = float(w)
                        h = float(h)

                        txtFileName = out_photo_prefix + 'Kinect_'+ str(kid+1)+ '/'+ str(frame_idx) + '.txt'
                        if not os.path.exists(os.path.dirname(txtFileName)):
                            try:
                                os.makedirs(os.path.dirname(txtFileName))
                            except OSError as exc:  # Guard against race condition
                                if exc.errno != errno.EEXIST:
                                    raise
                        with open(txtFileName, 'w') as f:
                            f.write(
                                str(0) + ' ' + str((x + w / 2) / img_w) + ' ' + str((y + h / 2) / img_h) + ' ' + str(
                                    w / img_w) + ' ' + str(
                                    h / img_h))
                        f.closed

            if img is None:
                break
            frame_idx += 1
            pass
    except Exception as e:
        print('Error: ', e)
        pass
    for i in range(7):
        depth_vids[i].close()
