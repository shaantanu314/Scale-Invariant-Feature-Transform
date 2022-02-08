from scipy import ndimage
import sys
import cv2

if __name__ == "__main__":
    
    args = sys.argv[1:]
    imgname = args[0]
    targetname = args[1]
    img = cv2.imread(imgname)
    rotated = ndimage.rotate(img, 45)
    cv2.imwrite(targetname,rotated)