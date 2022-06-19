import torch.utils.data as data
from PIL import Image
import numpy as np
from imageio import imread
from path import Path
import random
# import torch
import time
# import cv2
from PIL import ImageFile
# from scipy.misc import imresize
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_as_float(path):
    return imread(path).astype(np.float32)

def load_as_float_2(path):
    return np.array(Image.open(path),dtype=np.float32)

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    if(np.max(depth_png)<=255):
        print("max_value: ",np.max(depth_png))
        print("file_name: ",filename)
        assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    #depth[depth_png == 0] = -1.
    return depth

class SEMDataset(data.Dataset):
    def __init__(self, root, args, train=True, transform=None, transform_2=None, mode="DtoD"):
        np.random.seed(int(time.time()))                                                                    ## 랜덤하게 셔플하기 위한 장치
        random.seed(time.time())                                                                       ## 랜덤하게 셔플하기 위한 장치
        self.root = Path(root)                                                              ## Dataset의 경로 path설정
        self.depth_folder = self.root/'Train/Depth' if train else self.root/'Validation/Depth'
        self.img_folder = self.root/'Train/SEM' if train else self.root/'Validation/SEM'
        self.transform = transform
        self.transform_2 = transform_2  ## 매개변수로 받은 transform 클래스 대입
        self.train = train
        self.mode = mode
        self.args = args
        # self.resize = imresize
        self.crawl_folders()

        self.W = 64
        self.H = 32

    def crawl_folders(self):
        sequence_set = []                                                                 ## 처음 sequence_set을 빈 list로 초기화
        gt_depth = sorted(self.depth_folder.files('*.png'))
        rgbs = sorted(self.img_folder.files('*.png'))
        for i in range(len(rgbs)):
            tt = i//4
            sample = {'gt': gt_depth[tt], 'rgb': rgbs[i]}
            sequence_set.append(sample)
        if (self.args.img_test is False) or (self.train is True):
            random.shuffle(sequence_set)                                                            ## 모두 추가된 sequence_set를 셔플
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        gt_img = load_as_float(sample['gt'])
        rgb_img = load_as_float(sample['rgb'])

        if self.train is True:
            ## train_transform applied
            img_s = np.random.uniform(1, 1.2)
            scale = np.random.uniform(1.0, 1.5)
            gt_img = np.array(gt_img, dtype=np.float32)
            gt_img = np.resize(gt_img, (int(self.W), int( self.H)))
            rgb_img = np.array(rgb_img, dtype=np.float32)
            ## if RtoD train, color image get rotated and randomcrop also, but in DtoD mode only depth image transformed
            if self.args.mode == 'DtoD':
                rgb_img = np.resize(rgb_img, (self.W, self.H))
            else:
                rgb_img = np.resize(rgb_img, (int(img_s * self.W), int(img_s * self.H)))
            # gt_img = gt_img / scale
            if (gt_img.ndim == 2):
                gt_img_tmp = np.zeros((int(self.W), int(self.H), 1), dtype=np.float32)
                gt_img_tmp[:, :, 0] = gt_img[:, :]
                gt_img = gt_img_tmp
            if (rgb_img.ndim == 2):
                rgb_img_tmp = np.zeros((int(self.W), int(self.H), 1), dtype=np.float32)
                rgb_img_tmp[:, :, 0] = rgb_img[:, :]
                rgb_img = rgb_img_tmp

            if self.args.mode == 'DtoD':
                imgs_gt = self.transform([gt_img])
                imgs_rgb = self.transform([rgb_img])

                imgs_gt = np.resize(imgs_gt[0], (1, self.W, self.H))
                imgs_rgb = np.resize(imgs_rgb[0], (1, self.W, self.H))

                # imgs = [rgb_img] + imgs  ## [rgb, gt_depth]
            else:
                imgs = self.transform([rgb_img] + [gt_img])
                imgs[0] = np.resize(imgs[0], scale) #image
                imgs[1] = np.resize(imgs[1], scale) #depth

        else:
            ## valid_trasform applied
            gt_img = np.array(gt_img, dtype=np.float32)
            # gt_img = self.resize(gt_img,(251,340),'depth')
            ##gt_img = self.resize(gt_img,(251,340))
            rgb_img = np.array(rgb_img, dtype=np.float32)
            # rgb_img = self.resize(rgb_img,(251,340),'rgb')
            ##rgb_img = self.resize(rgb_img,(251,340))
            if (gt_img.ndim == 2):
                ##gt_img_tmp = np.zeros((251, 340 ,1),dtype=np.float32)
                gt_img_tmp = np.zeros((self.W, self.H, 1), dtype=np.float32)
                gt_img_tmp[:, :, 0] = gt_img[:, :]
                gt_img = gt_img_tmp
            imgs = self.transform([rgb_img] + [gt_img])
        # rgb_img = imgs[0]
        # gt_img = imgs[1]
        # rgb_img = np.resize(rgb_img, (1,self.W, self.H))


        return imgs_gt, imgs_rgb, gt_img

    def __len__(self):
        return len(self.samples)