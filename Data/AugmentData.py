import tkinter as tk
from Data.data_augmentation import DataAugmentation
from tkinter import filedialog


def main(argv=None):
    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(parent=root, initialdir="/", title="Please Select Image Folder Path")
    label_path = filedialog.askdirectory(parent=root, initialdir="/", title="Please Select Label Folder Path")
    save_path = filedialog.askdirectory(parent=root, initialdir="/", title="Please Select Save Folder Path")

    augmentation = DataAugmentation(data_path, label_path, save_path)
    augmentation.make_augmentation()


main()