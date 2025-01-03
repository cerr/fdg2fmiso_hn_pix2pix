# IMPORTANT
# here images are normalized to [-1, 1]



# from numpy.lib.type_check import real
# from posixpath import join
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import numpy as np
import torch
import os
import time
from scipy.io import savemat
from models.weighted_l1 import WeightedL1Loss

###helper function for save image
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1)/2 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.no_augmentation = True #no agumentation in testing
    opt.results_dir = os.path.join(opt.checkpoints_dir, 'best_model_test_result_image_no_augmentation')
    if not os.path.isdir(opt.results_dir):
        os.mkdir(opt.results_dir)
    if not os.path.isdir(os.path.join(opt.checkpoints_dir, 'best_model_test_result_mat_no_augmentation')):
        os.mkdir(os.path.join(opt.checkpoints_dir, 'best_model_test_result_mat_no_augmentation'))
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    opt.eval=True
    # # initialize logger
    # if opt.use_wandb:
    #     wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
    #     wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    # create txt file to store mae
    mae_file_name = os.path.join(opt.checkpoints_dir, 'best_model_mae_log_no_aug.txt')
    with open(mae_file_name, 'a') as file:
        now = time.strftime("%c")
        file.write('==========best model mae (tested at %s)==========\n' % now)
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        name = os.path.basename(data['B_paths'][0]).split('.')[0]
        real_A_tensor = visuals['real_A']
        real_B_tensor = visuals['real_B']
        fake_B_tensor = visuals['fake_B']
        # use -1 as tumor threshold as 0 transfered to -1 
        mask = torch.unsqueeze(real_A_tensor[:,1,:,:]>-1, 1)
        y_predict_tumor = fake_B_tensor[mask]
        y_target_tumor = real_B_tensor[mask]
        n_voxel = y_target_tumor.numel()
        # error = torch.abs(y_predict_tumor - y_target_tumor).sum()/n_voxel
        error = WeightedL1Loss(y_predict_tumor, y_target_tumor)
        message = '%s, %d, %.6f' % (name, n_voxel,error)
        with open(mae_file_name, 'a') as file:
            file.write('%s\n' % message)
        data_dict = {name + '_real_A' : real_A_tensor.cpu().numpy(), name + '_real_B' : real_B_tensor.cpu().numpy(), name + '_fake_B': fake_B_tensor.cpu().numpy()}
        savemat(opt.checkpoints_dir + '/best_model_test_result_mat_no_augmentation/' + name + '.mat', data_dict)
        # torch.save(real_A_tensor, opt.results_dir + '/' + name + '_real_A_tensor.pt')
        # torch.save(real_B_tensor, opt.results_dir + '/' + name + '_real_B_tensor.pt')
        # torch.save(fake_B_tensor, opt.results_dir + '/' + name + '_fake_B_tensor.pt')
    
        for label, img_tensor in visuals.items():
            if img_tensor.size(dim=1) == 2:
                img1 = tensor2im(torch.unsqueeze(img_tensor[:,0,:,:], 1))
                img2 = tensor2im(torch.unsqueeze(img_tensor[:,1,:,:], 1))
                img1_name = '%s_%s_1.png' % (name, label)
                img2_name = '%s_%s_2.png' % (name, label)
                save_path1 = os.path.join(opt.results_dir, img1_name)
                save_path2 = os.path.join(opt.results_dir, img2_name)
                save_image(img1, save_path1)
                save_image(img2, save_path2)
            else:
                img = tensor2im(img_tensor)
                img_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(opt.results_dir, img_name)
                save_image(img, save_path)


    #     img_path = model.get_image_paths()     # get image paths
    #     if i % 5 == 0:  # save images to an HTML file
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    # webpage.save()  # save the HTML