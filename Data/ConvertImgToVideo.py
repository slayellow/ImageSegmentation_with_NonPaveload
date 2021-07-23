import cv2
import glob
import tkinter
from tkinter import filedialog

img_array = []
size = 0

root = tkinter.Tk()
root.withdraw()
img_path = filedialog.askdirectory(parent=root, initialdir="/", title='Please Test Save Image Dir')
filename_list = []

for filename in glob.glob(img_path + "/*.png"):
    filename_list.append(filename)

filename_list.sort()

for filename in filename_list:
    img = cv2.imread(filename)
    print(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

print(size)
out = cv2.VideoWriter(img_path + "/output.avi", cv2.VideoWriter_fourcc(*'DIVX'), 50, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
