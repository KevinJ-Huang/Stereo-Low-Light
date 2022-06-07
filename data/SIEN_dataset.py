import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(input_left, target_left, input_right, target_right, patch_size, scale = 1, ix=-1, iy=-1):
    ih, iw, channels = input_left.shape
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    ipx = ip
    ipy = ip*4

    if ix == -1:
        ix = random.randrange(0, ih - ipx + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ipy + 1)

    # (tx, ty) = (scale * ix, scale * iy)


    input_left = input_left[ix:ix + ipx, iy:iy + ipy, :]  # [:, ty:ty + tp, tx:tx + tp]
    target_left = target_left[ix:ix + ipx, iy:iy + ipy, :]  # [:, iy:iy + ip, ix:ix + ip]
    input_right = input_right[ix:ix + ipx, iy:iy + ipy, :]
    target_right = target_right[ix:ix + ipx, iy:iy + ipy, :]

    return  input_left, target_left, input_right, target_right


def augment(input_left, target_left, input_right, target_right, hflip, rot):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot180 = rot and random.random() < 0.5

    def _augment(input_left, target_left, input_right, target_right):
        if hflip:
            input_left = input_left[:, ::-1, :]
            target_left = target_left[:, ::-1, :]
            input_right = input_right[:, ::-1, :]
            target_right = target_right[:, ::-1, :]
        if vflip:
            input_left = input_left[::-1, :, :]
            target_left = target_left[::-1, :, :]
            input_right = input_right[::-1, :, :]
            target_right = target_right[::-1, :, :]
        if rot180:
            input_left = cv2.rotate(input_left, cv2.ROTATE_180)
            target_left = cv2.rotate(target_left, cv2.ROTATE_180)
            input_right = cv2.rotate(input_right, cv2.ROTATE_180)
            target_right = cv2.rotate(target_right, cv2.ROTATE_180)

        return  input_left, target_left, input_right, target_right

    input_left, target_left, input_right, target_right = _augment(input_left, target_left, input_right, target_right)

    return input_left, target_left, input_right, target_right



# def get_image_hdr(img):
#     img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
#     img = np.round(img/(2**6)).astype(np.uint16)
#     img = img.astype(np.float32)/1023.0
#     return img

def get_image_ldr(img):
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
    return img


def load_image_train(group):
    # images = [get_image(img) for img in group]
    # inputs = images[:-1]
    # target = images[-1]
    input_left = get_image_ldr(group[0])
    target_left = get_image_ldr(group[1])
    input_right = get_image_ldr(group[2])
    target_right = get_image_ldr(group[3])
    # if black_edges_crop == True:
    #     inputs = [indiInput[70:470, :, :] for indiInput in inputs]
    #     target = target[280:1880, :, :]
    #     return inputs, target
    # else:
    return input_left, target_left,input_right,target_right


def transform():
    return Compose([
        ToTensor(),
    ])

def BGR2RGB_toTensor(input_left, target_left, input_right, target_right):
    input_left = input_left[:, :, [2, 1, 0]]
    target_left = target_left[:, :, [2, 1, 0]]
    input_right = input_right[:, :, [2, 1, 0]]
    target_right = target_right[:, :, [2, 1, 0]]
    input_left = torch.from_numpy(np.ascontiguousarray(np.transpose(input_left, (2, 0, 1)))).float()
    target_left = torch.from_numpy(np.ascontiguousarray(np.transpose(target_left, (2, 0, 1)))).float()
    input_right = torch.from_numpy(np.ascontiguousarray(np.transpose(input_right, (2, 0, 1)))).float()
    target_right = torch.from_numpy(np.ascontiguousarray(np.transpose(target_right, (2, 0, 1)))).float()

    return input_left, target_left,input_right,target_right



class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, upscale_factor, data_augmentation, group_file, patch_size, black_edges_crop, hflip, rot, transform=transform()):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.black_edges_crop = black_edges_crop
        self.hflip = hflip
        self.rot = rot

    def __getitem__(self, index):

        input_left, target_left, input_right, target_right = load_image_train(self.image_filenames[index])

        #target = target.resize((inputs.size[0], inputs.size[1]), Image.ANTIALIAS)

        # target = cv2.resize(target,(inputs.shape[1],inputs.shape[0]))
        # target = target[:768, :512]
        # inputs = inputs[:768, :512]

        if self.patch_size!=None:
            input_left, target_left, input_right, target_right = \
                get_patch(input_left, target_left, input_right, target_right , self.patch_size, self.upscale_factor)


        if self.data_augmentation:
            input_left, target_left, input_right, target_right = \
                augment(input_left, target_left, input_right, target_right, self.hflip, self.rot)

        if self.transform:
            input_left, target_left, input_right, target_right = \
                BGR2RGB_toTensor(input_left, target_left, input_right, target_right)


        return {'LQleft': input_left, 'GTleft': target_left, 'LQright': input_right, 'GTright': target_right, 'LQleft_path': self.image_filenames[index][0],
       'GTleft_path': self.image_filenames[index][1],'LQright_path': self.image_filenames[index][2], 'GTright_path': self.image_filenames[index][3]}

    def __len__(self):
        return len(self.image_filenames)


# if __name__ == '__main__':
#     output = 'visualize'
#     if not os.path.exists(output):
#         os.mkdir(output)
#     dataset = DatasetFromFolder(4, True, 'dataset/groups.txt', 64, True, True, True)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
#     for i, (inputs, target) in enumerate(dataloader):
#         if i > 10:
#             break
#         if not os.path.exists(os.path.join(output, 'group{}'.format(i))):
#             os.mkdir(os.path.join(output, 'group{}'.format(i)))
#         input0, input1, input2, input3, input4 = inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3], inputs[0][4]
#         vutils.save_image(input0, os.path.join(output, 'group{}'.format(i), 'input0.png'))
#         vutils.save_image(input1, os.path.join(output, 'group{}'.format(i), 'input1.png'))
#         vutils.save_image(input2, os.path.join(output, 'group{}'.format(i), 'input2.png'))
#         vutils.save_image(input3, os.path.join(output, 'group{}'.format(i), 'input3.png'))
#         vutils.save_image(input4, os.path.join(output, 'group{}'.format(i), 'input4.png'))
#         vutils.save_image(target, os.path.join(output, 'group{}'.format(i), 'target.png'))
