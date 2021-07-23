import tkinter as tk
from tkinter import filedialog
from os.path import getsize
import numpy as np
import os
import io
from PIL import Image

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir="./", title="Choose Image File",
                                       filetypes=(("image files", "*.triplecameraccdcenter"), ("all files", "*.*")))
save_path = filedialog.askdirectory(parent=root, initialdir="/", title="Please Select Save Folder Path")

file = open(file_path, 'rb')
one_packet_size = 921763                # One Packet Structure
file_size = getsize(file_path)
count = 0                               # Reading Count
image_shape = ( 640, 480 )
print("Extract Start ")
itr = 0
while count < file_size:

    file.seek(163, 1)                   # 라이다 데이터 이전 데이터들
    count = count + 163
    byte = file.read(921600)             # Lidar Layer Index
    count = count + 921600
    img = Image.frombytes('RGB', image_shape, byte)
    data = np.array(img)
    red, green, blue = data.T
    data = np.array([blue, green, red])
    data = data.transpose()
    img = Image.fromarray(data)
    img.save(save_path + "/data_{0:08d}.png".format(itr))
    print("data_{0:08d}.png".format(itr))
    itr += 1
file.close()
print('Convert RTheta To XYZ End!')


