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
# import time
from scipy.io import savemat
import torch.nn as nn
# import matplotlib.pyplot as plt

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
    opt.eval = True
    opt.results_dir = os.path.join(opt.checkpoints_dir, 'gradcam')
    if not os.path.isdir(opt.results_dir):
        os.mkdir(opt.results_dir)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
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

    class model_gradcam(nn.Module):
        def __init__(self):
            super(model_gradcam, self).__init__()
            self.model = model
            #get the input
            # self.down_1 = self.model.
            # self.down_2 = 
            # self.down_3 = 
            self.bn = nn.Sequential(
                self.model.netG.unet_block_3.model[0], 
                self.model.netG.unet_block_2.model[:3], 
                self.model.netG.unet_block_2.model[3].model[:3]
                )
            self.up_1 = nn.Sequential(
                self.model.netG.unet_block_3.model[0], 
                self.model.netG.unet_block_2.model[:5])
            self.up_2 = nn.Sequential(
                self.model.netG.unet_block_3.model[0], 
                self.model.netG.unet_block_2, 
                self.model.netG.unet_block_3.model[2])
            self.up_3 = self.model.netG.unet_block_3.model[:4]
            #placeholder for gradients
            self.gradients = None
        
        def activations_hook(self, grad):
            self.gradients = grad
        
        
        def forward(self, x0):
            x = self.up_2[:2](x0)
            # x = self.up_2(x0)
            # x = torch.max(x)
            # register the hook
            x.register_hook(self.activations_hook)

            # x = self.up_3(x0)
            # x.register_hook(self.activations_hook)
            # x, _ = torch.topk(x.flatten(), 1)
            # return torch.sum(x)
            x = self.up_3[2:](x)
            mask = torch.unsqueeze(x0[:,1,:,:], 1)>-1
            # x, _ = torch.topk(x[mask], 1)
            return torch.max(x[mask])
            # return torch.max(x)

        def get_activations_gradient(self):
            return self.gradients

        def get_activations(self, x):
            return self.up_2[:2](x)


    model_gradcam = model_gradcam()
    model_gradcam.eval()


    # create txt file to store mae
    # mae_file_name = os.path.join(opt.checkpoints_dir, 'best_model_mae_log.txt')
    # with open(mae_file_name, 'a') as file:
    #     now = time.strftime("%c")
    #     file.write('==========best model mae (tested at %s)==========\n' % now)
    #63: F3P6 Ferren primary slice 25 (number of tumor pixel 188), 
    #92: F3P9 Hadley lymph node slice 26
    #38: F4P4 Franchi lymph node slice 29
    #128: F4P11 Lustig primary slice 24

    # if os.path.basename(opt.dataroot)=='3':
    #     id_to_test = [63, 92]

    # if os.path.basename(opt.dataroot)=='4':
    #     id_to_test = [38, 128]
    # n_temp = 0
    # for i in range(268, 276):
    #     slice_temp = dataset.dataset[i]['A']
    #     if torch.sum(slice_temp[1,:,:]>-1) > n_temp:
    #         i_max = i
    #         n_temp = torch.sum(slice_temp[1,:,:]>-1)
    #         print(i)
    #         print(dataset.dataset.image_paths[i])
    id_to_test = [14, 188, 332, 270]

    for i, data in enumerate(dataset):
        if i in id_to_test:  # only apply our model to opt.num_test images.

            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            name = os.path.basename(data['B_paths'][0]).split('.')[0]
            real_A_tensor = visuals['real_A']
            real_B_tensor = visuals['real_B']
            fake_B_tensor = visuals['fake_B']
            # use -1 as tumor threshold as 0 transfered to -1 
            # mask = visuals['k1']>-1
            # y_predict_tumor = fake_B_tensor[mask]
            # y_target_tumor = real_B_tensor[mask]
            # n_voxel = y_target_tumor.numel()
            # error = torch.abs(y_predict_tumor - y_target_tumor).sum()/n_voxel
            # message = '%s, %d, %.6f' % (name, n_voxel,error)
            #get activations
            bn_act = model_gradcam.bn(data['A']).detach()
            up1_act = model_gradcam.up_1(data['A']).detach()
            up2_act = model_gradcam.up_2(data['A']).detach()
            up3_act = torch.tanh(model_gradcam.up_3(data['A'])).detach() #this is output
            #gradcam for activation of up_2 for sum of maximum three pixels in output of up_3 before tanh
            activation = model_gradcam.get_activations(data['A'])
            #get the id of max activation of up_3 within tumor
            # mask = torch.unsqueeze(data['A'][:,1,:,:], 1)>-1

            max_activation = model_gradcam(data['A'])
            max_activation.backward()
            gradients = model_gradcam.get_activations_gradient()
            pool_gradients = torch.mean(gradients, dim=[0, 2, 3])

            for ii in range(128):
                activation[:,ii,:,:] *= pool_gradients[ii]

            heatmap = torch.mean(activation, dim=1).squeeze().detach()

            heatmap = np.maximum(heatmap, 0)

            heatmap /= torch.max(heatmap)

            # plt.matshow(heatmap.squeeze())
            
            # with open(mae_file_name, 'a') as file:
            #     file.write('%s\n' % message)
            data_dict = {
                name + '_real_A' : real_A_tensor.cpu().numpy(), 
                name + '_real_B' : real_B_tensor.cpu().numpy(), 
                name + '_fake_B': fake_B_tensor.cpu().numpy(), 
                # name + '_k1': visuals['k1'].cpu().numpy(), 
                name + '_gradcam' : heatmap.cpu().numpy(),
                name + '_bn_act' : bn_act.cpu().numpy(),
                name + '_up1_act' : up1_act.cpu().numpy(),
                name + '_up2_act' : up2_act.cpu().numpy(),
                name + '_up3_act' : up3_act.cpu().numpy(),
                }
            savemat(opt.results_dir + '/' + name + '.mat', data_dict)
        # torch.save(real_A_tensor, opt.results_dir + '/' + name + '_real_A_tensor.pt')
        # torch.save(real_B_tensor, opt.results_dir + '/' + name + '_real_B_tensor.pt')
        # torch.save(fake_B_tensor, opt.results_dir + '/' + name + '_fake_B_tensor.pt')
    
        # for label, img_tensor in visuals.items():
        #     # if img_tensor.size(dim=1) == 2:
        #     #     img1 = tensor2im(torch.unsqueeze(img_tensor[:,0,:,:], 1))
        #     #     img2 = tensor2im(torch.unsqueeze(img_tensor[:,1,:,:], 1))
        #     #     img1_name = '%s_%s_1.png' % (name, label)
        #     #     img2_name = '%s_%s_2.png' % (name, label)
        #     #     save_path1 = os.path.join(opt.results_dir, img1_name)
        #     #     save_path2 = os.path.join(opt.results_dir, img2_name)
        #     #     save_image(img1, save_path1)
        #     #     save_image(img2, save_path2)
        #     if label != 'k1':
        #         img = tensor2im(img_tensor)
        #         img_name = '%s_%s.png' % (name, label)
        #         save_path = os.path.join(opt.results_dir, img_name)
        #         save_image(img, save_path)
