
import argparse
import os
from sklearn.model_selection import GroupKFold
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--foldNum', default=5, type=int, help='fold number for cross validation')
parser.add_argument('--trainDir', default='FDG_k1_patch_pair', help='directory containing train images')
args = parser.parse_args()
k = args.foldNum
fileDir = os.path.join('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO', args.trainDir)
_, _, filenames = next(os.walk(fileDir), (None, None, []))
filenames.sort()
patientName = [x.split('_')[1] for x in filenames]
patientNameUnique = list(set(patientName))
group = []
for x in patientName:
    group.append(patientNameUnique.index(x))

#copy files for k fold cross validation
gkf = GroupKFold(n_splits=k)
foldNum = 1
for train, test in gkf.split(patientName, groups=group):
    trainFileName = [filenames[i] for i in train]
    testFileName = [filenames[i] for i in test]
    dst = os.path.join(fileDir + '_' + str(k) + 'Fold', str(foldNum), 'train')
    os.makedirs(dst)
    for file in trainFileName:
        shutil.copy2(os.path.join(fileDir, file), dst)

    dst = os.path.join(fileDir + '_' + str(k) + 'Fold', str(foldNum), 'test')
    os.makedirs(dst)
    for file in testFileName:
        shutil.copy2(os.path.join(fileDir, file), dst)
    foldNum += 1






