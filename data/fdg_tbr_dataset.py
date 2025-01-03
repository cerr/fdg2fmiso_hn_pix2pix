"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset
#from numpy.lib.type_check import imag
# from data.image_folder import make_dataset
#from PIL import Image
from scipy.io import loadmat
import os 
import torch
#from skimage.transform import resize
import torchvision.transforms as transforms

class FdgTbrDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--no_augmentation', action='store_true', help='if specified, do not appy data augmentation')
        parser.set_defaults(input_nc=1, output_nc=1,
                            load_size=40, crop_size=32, display_winsize=32)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image file names
        full_dir = os.path.join(self.root, opt.phase)
        _, _, filenames = next(os.walk(full_dir), (None, None, []))
        filenames.sort()
        self.image_paths = [os.path.join(full_dir, filename) for filename in filenames]
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt, grayscale=True, method=Image.NEAREST, convert=False)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        AB_path = self.image_paths[index]    # needs to be a string
        img = loadmat(AB_path)
        key = list(img)[-1]
        img_nparray = img[key]
        # img_nparray_resize = resize(img_nparray, (256,256), order=0, preserve_range=True, anti_aliasing=False)
        # img_tensor = torch.from_numpy(img_nparray_resize)

        img_tensor = torch.from_numpy(img_nparray).permute(2, 0, 1)
        if self.opt.no_augmentation:
            img_tensor_return = img_tensor
        else:
            # apply_augmentation = transforms.Compose(
            #     [transforms.RandomCrop(self.opt.crop_size, padding=(self.opt.load_size - self.opt.crop_size)//2, fill=-1)])
            apply_augmentation = transforms.Compose([transforms.RandomRotation(180, fill=-1)])
            img_tensor_return = apply_augmentation(img_tensor)
        
        #data are expected in the format of [input_nc+output_nc, H, W]
        # data_A = (img_tensor_return[:self.opt.input_nc,:,:] if self.opt.direction == 'AtoB'
        #             else img_tensor_return[:self.opt.output_nc,:,:])   # needs to be a tensor
        # data_B = (img_tensor_return[self.opt.input_nc:,:,:] if self.opt.direction == 'AtoB'
        #             else img_tensor_return[self.opt.output_nc:,:,:])   # needs to be a tensor

        data_A = img_tensor_return[:self.input_nc,:,:]
        # data_B = img_tensor_return[self.input_nc + 1:,:,:]
        mask = img_tensor_return[self.input_nc:self.input_nc+1,:,:]
        #this is to facilitate the pix2pix model.py
        # print(f'image_tensor: {img_tensor_return.size()}')
        # print(f'size A: {data_A.size()}')
        # print(f'size B; {data_B.size()}')    
        # print(f'size k1; {k1.size()}') 

        # return {'A': data_A, 'B': data_B, 'A_paths': AB_path, 'B_paths': AB_path, 'k1': k1}
        return {'A': data_A, 'A_paths': AB_path, 'mask': mask}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)