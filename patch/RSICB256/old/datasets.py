import os
import os.path
import random

import torch.utils.data as data
from PIL import Image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = []
    classes_aux = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    for i in classes_aux:
         
          c = [d for d in os.listdir(dir+"/"+i) if os.path.isdir(os.path.join(dir+"/"+i, d))]
          print(i, ": ", c) 
          classes += c
        
    
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

#def split_train_test(labels, data_dir='/home/smoreno/fedml_data/RSI-CB256'):
    ## Collect all image paths
    #image_paths = []
    #for i in range(len(category_names)):

        ## load all images per class
        #for j in range(N_images):        
            #img_path = os.path.join(data_dir, category_names[i], category_names[i]+'{:02d}.tif'.format(j))

            #image_paths.append(img_path)

    ## Split data into training and test
    #image_paths = np.array(image_paths)

    #nb_images = image_paths.shape[0]
    #shuffled_indices = np.arange(nb_images)
    #np.random.seed(seed)
    #np.random.shuffle(shuffled_indices)

    #training_indices = shuffled_indices[:int(0.8*nb_images)]               # 70% for training
    #test_indices = shuffled_indices[int(0.8*nb_images):]                   # 20% for test

    #train_paths = image_paths[training_indices]
    #test_paths = image_paths[test_indices]

    ## Copy images to train, val, test directories

    ## make directories:
    #for split in ['train', 'test']:
        #if not os.path.exists('data/RSICB256/'+split):
            #os.makedirs('data/RSICB256/'+split)

    #def copy_images_to_split_directory(paths, split):
        #for p in paths:
            #dst=p.replace('RSICB256/Images', 'RSICB256/'+split)
            #if not os.path.exists(os.path.dirname(dst)):
                #os.makedirs(os.path.dirname(dst))
            #copyfile(src=p, dst=dst)

def make_dataset(dir, class_to_idx, extensions, classes, train=True, trper=0.8, seed=123):
    images = []
    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)
    
    for auxdir in os.listdir(dir):
        auxdir2 = os.path.join(dir, auxdir)
        for target in sorted(os.listdir(auxdir2)):
            d = os.path.join(auxdir2, target)
            #if not os.path.isdir(d):
                #continue

            target_num = 0
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target], target)
                        images.append(item)
                        target_num += 1

            #net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
            #data_local_num_dict[class_to_idx[target]] = target_num
            #sum_temp += target_num

    random.seed(seed)
    random.shuffle(images)
    #print(sorted(images, key=lambda x: x[1])); exit()
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
        #net_dataidx_map[class_to_idx[count_class_num]] = (sum_temp, sum_temp + target_num)
        
    start, end = 0, 0
    class_index_aux = 0
    data = sorted(data, key=lambda x: x[1])
    for i in data:
        class_num = i[1]
        count_class_num = i[2]
        class_index = class_to_idx[count_class_num]
        if class_index != class_index_aux:
            start = end 
            class_index_aux = class_index
        end += 1
        net_dataidx_map[class_to_idx[count_class_num]] = (start, end)
            
        
    print(len(data), data_local_num_dict, net_dataidx_map)

    #trsize  = int(len(images)*trper)
    #images = images[:trsize] if train == True else images[trsize:]
    #print(len(images), data_local_num_dict, net_dataidx_map)
    #print(train)
    #print(len(tr_images), len(te_images))
    #assert len(images) == sum_temp
    
    return images, data_local_num_dict, net_dataidx_map


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage  # pylint: disable=E0401

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class RSICB256(data.Dataset):
    def __init__(
        self,
        data_dir,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        Generating this class too many times will be time-consuming.
        So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.loader = default_loader
        self.data_dir = "/home/smoreno/fedml_data/RSI-CB256"

        (
            self.all_data,
            self.data_local_num_dict,
            self.net_dataidx_map,
        ) = self.__getdatasets__()
        
        if dataidxs == None:
            self.local_data = self.all_data
        elif type(dataidxs) == int:
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin:end]
        else:
            self.local_data = []
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin:end]

    def get_local_data(self):
        return self.local_data

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(data_dir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.data_dir)
        IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
        all_data,data_local_num_dict, net_dataidx_map = make_dataset(
            self.data_dir, class_to_idx, IMG_EXTENSIONS, len(classes), self.train
        )
        if len(all_data) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.data_dir + "\n"
                    "Supported extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )
        return all_data, data_local_num_dict, net_dataidx_map

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)


class RSICB256_truncated(data.Dataset):
    def __init__(
        self,
        imagenet_dataset: RSICB256,
        dataidxs,
        net_dataidx_map,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.net_dataidx_map = net_dataidx_map
        self.loader = default_loader
        self.all_data = imagenet_dataset.get_local_data()
        #print("wwww", dataidxs)
        if dataidxs == None:
            self.local_data = self.all_data
        elif type(dataidxs) == int:
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin:end]
        else:
            self.local_data = []
            #print("--", dataidxs)
            for idxs in dataidxs:
                #print(idxs)
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin:end]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)
