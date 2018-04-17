import os, numpy as np, matplotlib.pyplot as plt, cv2

root_dir = '/media/data/datasets/Kinect2017-10/Datasets/'
out_dir = '/home/nguyenductrung/kinect_vis/output/'


# out_dir='/media/data/datasets/Kinect_vis/v2/'
def read_skeleton(path):
    data = {}
    f = open(path, 'rt')
    while True:
        line = f.readline()
        if line == '':
            break
        frame_idx = int(line.split(' ')[2])
        num_ppl = int(line.split(' ')[3])

        skels = []
        for i in range(num_ppl):
            line = f.readline().split(' ')
            skel = []
            for j in range(20):
                skel.append([int(float(line[j * 3])), int(float(line[j * 3 + 1]))])
            skels.append(skel)
        # print skel
        # break
        data[frame_idx] = skels

    # print line
    return data


# for _, dirs, _ in os.walk(root_dir):
#     break
dirs = ['20171123_Phong_lan2_23-11-2017__11-49-53']
print dirs
for f_name in dirs:
    # f_name='20171123_Hung_lan1_23-11-2017__11-05-57'
    vid_dir = root_dir + f_name + '/'
    out_vid_path = out_dir + f_name + '.avi'
    if os.path.isfile(out_vid_path):
        continue
    print f_name


    def read_acc(file_name):
        f = open(vid_dir + file_name, 'rt')
        lines = f.read().strip().split('\n')
        acc_hand = []
        # print lines[:10]
        for l in lines:
            row = [float(v) for v in l.split(' ')]
            if acc_hand == [] or (row[1] > acc_hand[-1][1]) or row[1] == 0:
                # print row
                acc_hand.append(row)

        # print len(acc_hand)
        start_timestamp = acc_hand[0][0]
        end_timestamp = acc_hand[-1][0]

        duration = end_timestamp - start_timestamp  # ms
        # print duration
        acc_new = np.zeros((int(duration) / 10 + 1, 3), dtype=float)

        for acc in acc_hand:
            time_diff = int(acc[0] - start_timestamp) / 10
            # print time_diff
            # print acc[1]
            acc_new[time_diff, 0] = acc[2]
            acc_new[time_diff, 1] = acc[3]
            acc_new[time_diff, 2] = acc[4]
        return acc_new


    import struct


    # from scipy.ndimage.interpolation import shift

    def read_depth(file_reader):
        try:
            byte = file_reader.read(4)
            val = struct.unpack('=L', byte)
            # print 'img size: '+str(val[0])
            img_data = np.fromstring(file_reader.read(val[0]), dtype=np.uint8)
            depth = cv2.imdecode(img_data, cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (640 / 3, 480 / 3))
            # depth = np.right_shift(depth, 7)
            # depth = np.asarray(depth, np.uint8)
            depth = np.left_shift(depth, 5)

            # depth = 255 - depth
            # ret, depth = cv2.threshold(depth, 254, 255, cv2.THRESH_TOZERO_INV)

            ret, mask = cv2.threshold(depth, 1, 255, cv2.THRESH_BINARY)

            # depth_vis = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            # depth
            # return depth_vis
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            return mask
        except Exception as e:
            print 'Error: ' + str(e)
            return None


    # each row is data for 10 ms
    acc_hand = read_acc('1.txt')
    acc_belt = read_acc('155.txt')

    # print acc_hand[:100]

    duration = acc_hand.shape[0] / 100.0  # s
    n_frames = int(duration * 20.0)  # each frame is 50ms

    pixels_per_sec = 50  # width
    pixels_per_ms = pixels_per_sec / 1000.0  # width
    length_acc_on_img = 1180 / 50  # 10s

    color_vids = []
    depth_vids = []
    skeleton_all = []
    for i in range(1, 8):
        vid = cv2.VideoCapture(vid_dir + '/Kinect_' + str(i) + '/color.avi')
        color_vids.append(vid)
        print vid_dir + 'Kinect_' + str(i) + '/depth.bin'
        f = open(vid_dir + 'Kinect_' + str(i) + '/depth.bin', "rb")
        depth_vids.append(f)
        skeleton_all.append(read_skeleton(vid_dir + '/Kinect_' + str(i) + '/skeleton.txt'))

    writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'H264'), 20.0, (1920, 1440))

    frame_idx = 0
    # cv2.namedWindow('',cv2.WINDOW_NORMAL)
    try:
        while True:
            if frame_idx % 10 == 0:
                print frame_idx
            img_total = np.zeros((480 * 3, 640 * 3, 3), np.uint8)
            for kid in range(7):
                ret, img = color_vids[kid].read()
                if img is None:
                    break
                depth = read_depth(depth_vids[kid])

                temp, contours, hierarchy = cv2.findContours(depth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                depth = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(depth, contours, -1, (0, 255, 0), 1)

                maxContourIndex = 0
                if (len(contours) > 0):
                    for index, contour in enumerate(contours):
                        if (cv2.contourArea(contour) >= cv2.contourArea(contours[maxContourIndex])):
                            maxContourIndex = index
                    x, y, w, h = cv2.boundingRect(contours[maxContourIndex])
                    cv2.rectangle(depth, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.rectangle(img, (x*3, y*3), (x*3 + w*3, y*3 + h*3), (0, 255, 0), 3)


                # print 'Color shape is:'+str(img.shape)
                # print 'depth shape is:'+str(depth.shape)
                location_x = kid % 3 * 640
                location_y = kid / 3 * 480

                img_total[location_y:location_y + 480, location_x:location_x + 640, :] = img
                img_total[location_y:location_y + 160, location_x:location_x + 213, :] = depth

                cv2.putText(img_total, str(kid + 1), (location_x + 10, location_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 5, cv2.LINE_AA)
            # print depth.shape
            # break
            if img is None:
                break
            img = img_total
            writer.write(img)
            # cv2.namedWindow('',cv2.WINDOW_NORMAL)
            # cv2.imshow('',img)
            # if cv2.waitKey(10)==27:
            # 	break
            # break
            # pass

            # cv2.imshow('',img)
            # cv2.waitKey(10)
            frame_idx += 1
            pass
    except Exception as e:
        pass
    for i in range(7):
        depth_vids[i].close()

# writer.close()