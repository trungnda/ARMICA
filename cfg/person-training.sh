TRAIN
./darknet detector train build/darknet/x64/data/person.data build/darknet/x64/data/yolov2-person.cfg build/darknet/x64/darknet19_448.conv.23
TEST
./darknet detector test build/darknet/x64/data/person.data cfg/yolov2-person.cfg backup/yolov2-person_300.weights

