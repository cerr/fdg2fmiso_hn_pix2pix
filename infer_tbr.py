''' Compute TBR from the input FDG scan and GTV segmentation
Inputs: Path to directory containing FDG PET scan and corresponding GTV segmentation.
        FDG scan must be in NifTi format and named as fdg_scan.nii.gz
        GTV segmentation must be in NifTi format and named as gtv_seg.nii.gz

'''

# pre-processing steps for running the inference
# FDG PET images is normalized to [-1, 1]
# normalization max value: FDG_max=30.1237, TBR_max=4.7117
# interploation of image to in-plane 1.95mm x 1.95mm voxel size, for each axial slice containing GTV.
# crop 32 x 32 x n slices enclosing GTV with its center the same as GTV center,
# then store the FDG and mask as 32 x 32 x 2 matrix.
# linear interploation for FDG, nearest interpolation for mask
# normalize image to [-1, 1] ptFDG = 2*ptFDG/FDG_max - 1;ptTBR = 2*ptTBR/TBR_max - 1;mask = 2*mask/mask_max - 1 (mask_max=1)


from models import create_model
import numpy as np
import torch
import os
import sys
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.radiomics.preprocess import getResampledGrid, imgResample3D
from cerr.utils.ai_pipeline import getScanNumFromIdentifier
from cerr.utils.image_proc import resizeScanAndMask
from cerr.utils.mask import computeBoundingBox

# Define FDG and TBR max values for normalization
FDG_max = 30.1237
TBR_max = 4.7117

# Pre-processing routine
def process_image(planC):
    """
    Resample FDG PET scan to 1.95mm x 1.95mm in-plane and resize to 32x32 voxels
    """

    modality = 'PT'
    identifier = {'imageType': 'PT SCAN'}

    grid_type = 'center'
    resamp_method = 'sitkBSpline'
    output_res = [0.195, 0.195, 0]  # Output res: 1mm x 1mm in-plane

    resize_method = 'pad2d'
    out_size = [32, 32]

    # Get scan array
    scan_num = getScanNumFromIdentifier(identifier, planC)[0]
    x_vals, y_vals, z_vals = planC.scan[scan_num].getScanXYZVals()
    scan_arr = planC.scan[scan_num].getScanArray()

    # Extract GTV outline
    outline_mask_arr = rs.getStrMask(0, planC)

    # Resample
    [x_resample, y_resample, z_resample] = getResampledGrid(output_res,
                                                            x_vals, y_vals, z_vals,
                                                            grid_type)
    resamp_scan_arr = imgResample3D(scan_arr,
                                     x_vals, y_vals, z_vals,
                                     x_resample, y_resample, z_resample,
                                     resamp_method, inPlane=True)
    resamp_mask_arr = imgResample3D(outline_mask_arr.astype(float),
                                     x_vals, y_vals, z_vals,
                                     x_resample, y_resample, z_resample,
                                     resamp_method, inPlane=True)
    resamp_mask_arr = resamp_mask_arr.astype(int)
    resample_grid = [x_resample, y_resample, z_resample]
    planC = pc.importScanArray(resamp_scan_arr,
                                   resample_grid[0], resample_grid[1], resample_grid[2],
                                   modality, scan_num, planC)
    resample_scan_num = len(planC.scan) - 1

    # Crop to GTV outline on each slice
    sum_slices = np.sum(resamp_mask_arr, axis=(0, 1))
    valid_slices = np.where(sum_slices > 0)[0]
    num_slices = len(valid_slices)
    limits = np.zeros((num_slices, 4))

    for slc in range(num_slices):
        minr, maxr, minc, maxc, _, _, _ = computeBoundingBox( \
                resamp_mask_arr[:, :, valid_slices[slc]],
                is2DFlag=True)
        limits[slc, :] = [minr, maxr, minc, maxc]

    rowMin = np.min(limits[:,0])
    rowMax = np.max(limits[:,1])
    colMin = np.min(limits[:,2])
    colMax = np.max(limits[:,3])

    limits[:,0] = rowMin
    limits[:,1] = rowMax
    limits[:,2] = colMin
    limits[:,3] = colMax

    # Resize to 32 x 32 in-plane
    resamp_slc_arr = resamp_scan_arr[:, :, valid_slices]
    slc_grid = (resample_grid[0], resample_grid[1], resample_grid[2][valid_slices])
    resamp_mask_arr = resamp_mask_arr[:,:,valid_slices,np.newaxis].astype(int)
    proc_scan_arr, mask_out_4d, resize_grid = resizeScanAndMask(resamp_slc_arr,
                                                               resamp_mask_arr,
                                                               slc_grid,
                                                               out_size,
                                                               resize_method,
                                                               limitsM=limits)

    resamp_mask_arr = resamp_mask_arr[:,:,:,0]
    resampGrid = (x_resample, y_resample, z_resample)
    return proc_scan_arr, mask_out_4d, resize_grid, limits, resamp_mask_arr, resampGrid, valid_slices, planC

# Routine for inference
def main(argv):
    ''' applies inference to the input FDG scan and writes output TBR volume in the output directory
    '''

    input_nii_path = argv[1]
    output_path = argv[2]
    os.makedirs(output_path, exist_ok=True)

    scriptDir = os.path.dirname(os.path.abspath(__file__))
    checkpointsDir = os.path.join(scriptDir, "checkpoints")

    # Create output dir, if required
    os.makedirs(output_path, exist_ok=True)

    fmisoOutputFile = os.path.join(output_path, 'tbr_pred.nii.gz')

    from argparse import Namespace
    opt = Namespace(aspect_ratio=1.0, batch_size=1,
              checkpoints_dir=checkpointsDir, crop_size=32,
              dataroot=input_nii_path, dataset_mode='fdg_tbr',
              direction='AtoB', display_winsize=32, epoch='latest', eval=False, gpu_ids='-1', init_gain=0.02,
              init_type='normal', input_nc=1, load_iter=0, load_size=40, max_dataset_size=np.inf,
              model='pix2pix_32_fdg_tbr_three_down_layer_four_by_four', n_layers_D=1, name='fdg_tbr_pix2pix', ndf=64,
              netD='basic', netG='unet_32', ngf=64, no_augmentation=False, no_dropout=False, no_flip=False,
              norm='instance', num_test=50, num_threads=4, output_nc=1, phase='test', preprocess='resize_and_crop',
              results_dir=output_path, serial_batches=False, suffix='', use_wandb=False, verbose=False,
              isTrain=False)

    #opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.no_augmentation = True
    opt.gpu_ids = []

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    opt.eval = True

    if opt.eval:
        model.eval()

    # Find scan and segmentation files
    ptScanFile = ''
    gtvSegFile = ''
    for root, dirs, files in os.walk(input_nii_path):
        for file in files:
            if 'fdg_scan' in file.lower():
                ptScanFile = os.path.join(root, file)
            if 'gtv_seg' in file.lower():
                gtvSegFile = os.path.join(root, file)

    # Import NIfTI scan to planC
    planC = pc.loadNiiScan(ptScanFile, imageType="PT SCAN")
    planC = pc.loadNiiStructure(gtvSegFile, 0, planC, {1: 'GTV'})

    # Zero-out volume outside the GTV as done for training
    scan_vol, mask_vol, resize_grid, limits, resamp_mask_arr, resampGrid, valid_slices, planC = process_image(planC)
    mask_vol = mask_vol[:,:,:,0].astype(float)
    scan_vol[mask_vol < 1] = 0

    vol_shape = scan_vol.shape
    num_slices = vol_shape[2]
    input_size = vol_shape[:2]
    np.clip(scan_vol, 0, FDG_max, out=scan_vol)
    norm_scan = 2*scan_vol/FDG_max - 1
    norm_mask = 2*mask_vol - 1
    tbr_pred = np.zeros((input_size[0], input_size[1], num_slices))
    for i in range(num_slices):
        scan_slice = np.transpose(norm_scan[:,:,i].astype('<f4'))
        mask_slice = np.transpose(norm_mask[:,:,i].astype('<f4'))
        zero_slice = -1 * np.ones(scan_slice.shape,dtype='<f4')
        scan_and_mask = np.stack((scan_slice,mask_slice,zero_slice),axis=2) # 32x32x3
        img_tensor = torch.from_numpy(scan_and_mask).permute(2, 0, 1) # 3x32x32
        data_A = img_tensor[0, :, :]
        data_A = data_A[np.newaxis,np.newaxis,:,:]
        mask = img_tensor[1, :, :]
        mask = mask[np.newaxis,np.newaxis,:,:]
        data = {'A': data_A, 'A_paths': '', 'mask': mask}

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        real_A_tensor = visuals['real_A']
        fake_B_tensor = visuals['fake_B']
        tbr = fake_B_tensor.cpu().numpy()
        tbr_pred[:,:,i] = np.transpose(tbr[0,0,:,:])

    tbr_pred = TBR_max / 2 * (tbr_pred + 1)

    # Resample TBR scan to the original scan resolution
    from cerr.radiomics.preprocess import imgResample3D
    fdgScanNum = 0
    xFDG, yFDG, zFDG = planC.scan[fdgScanNum].getScanXYZVals()
    resampMethod = 'sitkLinear'
    extrapVal = 0
    resampTBR = imgResample3D(tbr_pred,
                                     resize_grid[0][:,0], resize_grid[1][:,0], resize_grid[2],
                                     xFDG, yFDG, zFDG,
                                     resampMethod, extrapVal)

    # Add TBR to planC
    planC = pc.importScanArray(resampTBR, xFDG, yFDG, zFDG, 'TBR SCAN', fdgScanNum, planC)

    # Export TBR scan to Nii
    tbrScanNum = len(planC.scan) - 1
    planC.scan[tbrScanNum].saveNii(fmisoOutputFile)

    return 0


if __name__ == '__main__':
    main(sys.argv)

