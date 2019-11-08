import cv2
import sys

args = sys.argv
print('open ' + args[1])
vidcap = cv2.VideoCapture(args[1])
success,image = vidcap.read()
count = 0
while success:
    filename = '../data/bmp/' + args[1].split('/')[-1].split('.')[0] + '/frame' + str(count).zfill(6) + '.bmp'
    cv2.imwrite(filename, image)
    success,image = vidcap.read()
    print(count)
    count += 1