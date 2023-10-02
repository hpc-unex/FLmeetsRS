import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import random
#from torchvision.datasets import RSICB256

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def default_loader(path):
    return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class RSICB256_truncated(data.Dataset):
    def __init__(
        self, dir, dataidxs=None, train=True, transform=None, target_transform=None, download=False, args=None,
    ):
        self.args = args
        self.dir = "~/FLmeetsRS/FedML/fedml_data/RSICB256"
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.loader = default_loader
        self.data, self.target, self.path = self.__build_truncated_dataset__()
        
    def has_file_allowed_extension(self,filename, extensions):
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)
        
    def find_classes(self):
        classes = []
        classes_aux = [d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))]
        for i in classes_aux:
            
            c = [d for d in os.listdir(self.dir+"/"+i) if os.path.isdir(os.path.join(self.dir+"/"+i, d))]
            #print(i, ": ", c) 
            classes += c
            
        
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    def make_dataset(self, class_to_idx, extensions, classes, train=True, trper=None, seed=123):
        images = []
        
        data_local_num_dict = dict()
        net_dataidx_map = dict()
        sum_temp = 0
        dir = os.path.expanduser(self.dir)
        
        for auxdir in os.listdir(dir):
            auxdir2 = os.path.join(dir, auxdir)
            for target in sorted(os.listdir(auxdir2)):
                d = os.path.join(auxdir2, target)

                target_num = 0
                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if self.has_file_allowed_extension(fname, extensions): #and target_num < 100:
                            path = os.path.join(root, fname)
                            pathv = Image.open(os.path.join(root, fname))
                            img = np.array(pathv)
                            item = (pathv, class_to_idx[target], target, path)
                            images.append(item)
                            target_num += 1
                            pathv.close()

        random.seed(seed)
        random.shuffle(images)
        size  = int(len(images)*trper)
        data = []
        target_num = []
        for target in range (0,classes):
            target_num.append(0)
        
        if train:        
            images = images[:size]
        else:
            images = images[size:]
            
        for i in images:
            class_num = i[1]
            count_class_num = i[2]
            data.append(i)
            target_num[class_num] += 1
            data_local_num_dict[class_to_idx[count_class_num]] = target_num[class_num]
        return data

    def __build_truncated_dataset__(self):

        #cifar_dataobj = RSICB256(self.root, self.train, self.transform, self.target_transform, self.download)
        classes, class_to_idx = self.find_classes()
        IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
        images = self.make_dataset(
             class_to_idx, IMG_EXTENSIONS, len(classes), self.train, self.args.tr_percent
        )
        #print(images[0])
        a = np.array(images)
        data = a[:,0]
        target = a[:,1]
        path = a[:,3]
        
        #print(target)
        #exit()
        
        #data = cifar_dataobj.data
        #target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            path = path[self.dataidxs]

        return data, target, path

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, path = self.data[index], self.target[index], self.path[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)
