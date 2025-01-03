
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
# parser.add_argument('--foldNum', default=5, type=int, help='fold number for cross validation')
#path contains fold separation to be reproduce
parser.add_argument('--sourceDir', default='FDGk1_k3_patch_pair_k3_log_transformed_5Fold', help='directory containing fold spearation to be reproduced')
parser.add_argument('--destDir', default='FDGk1_TBR_patch_pair', help='directory containing data be to separated accroding to sourceDir')
args = parser.parse_args()
sourceDir = os.path.join('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO', args.sourceDir)
destDir = os.path.join('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO', args.destDir)
foldFolders = os.listdir(sourceDir)
foldNum = len(foldFolders)

for i in range(1,foldNum+1):
    trainFoldDir = os.path.join(sourceDir, str(i), 'train')
    _, _, filenames = next(os.walk(trainFoldDir), (None, None, []))
    dst = os.path.join(destDir + '_' + str(foldNum) + 'Fold', str(i), 'train')
    os.makedirs(dst)
    for fname in filenames:
        # shutil.copy2(os.path.join(destDir, fname), dst)
        # if file names in source and destation are the same, then use the line above
        #here we need to add 'nolog' to file name
        fname_dest = fname[:5] + 'TBR' + fname[7:]
        shutil.copy2(os.path.join(destDir, fname_dest), dst)
    testFoldDir = os.path.join(sourceDir, str(i), 'test')
    _, _, filenames = next(os.walk(testFoldDir), (None, None, []))
    dst = os.path.join(destDir + '_' + str(foldNum) + 'Fold', str(i), 'test')
    os.makedirs(dst)
    for fname in filenames:
        # shutil.copy2(os.path.join(destDir, fname), dst)
        # if file names in source and destation are the same, then use the line above
        #here we need to add 'nolog' to file name
        fname_dest = fname[:5] + 'TBR' + fname[7:]
        shutil.copy2(os.path.join(destDir, fname_dest), dst)        








