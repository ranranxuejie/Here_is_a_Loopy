import os
import xml.etree.ElementTree as ET

from tqdm import tqdm
from pathlib import Path
import random
from shutil import copyfile

Source_root = Path('/loopy_detect_YOLOv8\dataprocess\DATA') # 源文件夹

Target_root = Path('/loopy_detect_YOLOv8\dataprocess\YOLO') # 目标文件夹
classes = ['loopy'] # 仅分类一种，还可自定义多种类别（需有对应数据集）

def insulatorDataSet2YOLO(Source_root, Target_root, is_test, shuffle, train_ratio):
    '''
    Source_root:为原始数据集的目录
    Target_root:为转化为YOLO形式的目录
    is_test:是否划分测试集
    shuffle:是否对数据进行打乱
    train_ratio:训练集的比例
    '''
    def make_folders(base_folder, subfolders):
        for subfolder in subfolders:
            folder_path = os.path.join(base_folder, subfolder)
            os.makedirs(folder_path, exist_ok=True)

    os.makedirs(Target_root, exist_ok=True)
    images_folder = os.path.join(Target_root, 'images')
    labels_folder = os.path.join(Target_root, 'labels')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    image_subfolders = ['train', 'val', 'test'] if is_test else ['train', 'val']
    label_subfolders = ['train', 'val', 'test'] if is_test else ['train', 'val']

    make_folders(images_folder, image_subfolders)
    make_folders(labels_folder, label_subfolders)

    source_im_folder = os.path.join(Source_root, 'images')
    target_lb_folder = os.path.join(Source_root, 'labels')

    images = [f for f in os.listdir(source_im_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_images = len(images)

    train_size = int(train_ratio * total_images)
    test_size = int((1.0-train_ratio) * total_images)

    if is_test:
        test_size /= 2.0
        val_size = total_images - train_size - test_size
        splits = {'train': train_size, 'val': val_size, 'test': test_size}
    else:
        splits = {'train': train_size, 'val': test_size}
    if shuffle:
        random.shuffle(images)

    start_idx = 0
    for split, size in splits.items():
        split_images = images[start_idx:(start_idx + size)]
        start_idx += size
        copy_images(source_im_folder, target_lb_folder, split_images, split)

def copy_images(source_folder_im, source_folder_lb, image_list, split):

    for image in image_list:
        # 复制图像文件
        source_image_path = os.path.join(source_folder_im, image)
        dest_image_path = os.path.join(Target_root, 'images', split, image)
        copyfile(source_image_path, dest_image_path)

        id = os.path.splitext(image)[0]
        annotation_file = image.replace('.jpg', '.txt')
        dest_label_path = os.path.join(Target_root, 'labels', split, annotation_file)
        convert_label(Source_root, dest_label_path, id)


def convert_label(path, lb_path, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh
    for class_name in classes:
        # in_file = open(path / f'labels/{class_name}/{image_id}.xml')
        in_file = open(path / f'labels/{image_id}.xml')
        out_file = open(lb_path, 'a')

        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        # print(w,h)
        names = list(['loopy_detect_YOLOv8'])  # names list
        for obj in root.iter('object'):
            cls = obj.find('name').text
            # if cls in names and int(obj.find('difficult').text) != 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + '\n')

insulatorDataSet2YOLO(Source_root, Target_root, is_test=False, shuffle=True, train_ratio=0.8)
