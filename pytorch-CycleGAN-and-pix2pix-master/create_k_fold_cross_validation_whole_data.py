
import argparse
import os
from sklearn.model_selection import StratifiedGroupKFold
import shutil
import time

parser = argparse.ArgumentParser()
parser.add_argument('--foldNum', default=10, type=int, help='fold number for cross validation')
parser.add_argument('--trainDir', default='FDGk1_TBR_patch_pair', help='directory containing train images')
args = parser.parse_args()
k = args.foldNum
oldDataDir = os.path.join('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO', args.trainDir)
newDataDir = os.path.join('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/new_data', args.trainDir)
_, _, filenames_old = next(os.walk(oldDataDir), (None, None, []))
filenames_old.sort()
_, _, filenames_new = next(os.walk(newDataDir), (None, None, []))
filenames_new.sort()
#get the primary/LN in txt file for the old data as the filenames_old does not contain such information 
with open('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/old_data.txt') as f:
    lines = f.readlines()
pName, label = zip(*(s.split() for s in lines))
patientNameOld = [x.split('_')[1] if len(x.split('_')) == 3 else '_'.join(x.split('_')[1:3]) for x in filenames_old]
#use primary/LN as lable for stratified separation of folds
label_old = [1 if 'primary' in label[pName.index(name)].casefold() else 0 for name in patientNameOld]
label_new = [1 if 'primary' in x.casefold() else 0 for x in filenames_new]

#combined
whole_dir = '/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO/FDGk1_TBR_patch_pair_whole'
_, _, filenames_combined = next(os.walk(whole_dir), (None, None, []))
filenames_combined.sort()
label_combined = [label_old[filenames_old.index(x)] if x in filenames_old else label_new[filenames_new.index(x)] for x in filenames_combined]
patientName_combined = [x.split('_')[1] for x in filenames_combined]
patientNameUnique_combined = list(set(patientName_combined))
group = []
for x in patientName_combined:
    group.append(patientNameUnique_combined.index(x))

#copy files for k fold cross validation
# cv = StratifiedGroupKFold(n_splits=k)
# foldNum = 1
# for train_idx, test_idx in cv.split(filenames_combined, label_combined, group):
#     trainFileName = [filenames_combined[i] for i in train_idx]
#     testFileName = [filenames_combined[i] for i in test_idx]
#     dst = os.path.join(whole_dir + '_' + str(k) + 'Fold', str(foldNum), 'train')
#     os.makedirs(dst)
#     for file in trainFileName:
#         shutil.copy2(os.path.join(whole_dir, file), dst)

#     dst = os.path.join(whole_dir + '_' + str(k) + 'Fold', str(foldNum), 'test')
#     os.makedirs(dst)
#     for file in testFileName:
#         shutil.copy2(os.path.join(whole_dir, file), dst)
#     foldNum += 1

#summary of patients, primary and lymph nodes in train, test, dev
log_file_name = os.path.join('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO', 'fold_separation_info.txt')
with open(log_file_name, 'w') as file:
    now = time.strftime("%c")
    file.write('==========patient, priamry/LN, fold, id in fold, primary (1)/LN (0) (%s)==========\n' % now)


##70 train 10 dev 20 test split according to the separation of 10 folds
train_folds = [2, 3, 4, 7, 8, 9, 10]
dev_folds = [6]
test_folds = [1 , 5]
dst_train = os.path.join(whole_dir + '_70_10_20', 'train')
os.makedirs(dst_train)
for i in train_folds:
    fold_path = os.path.join(whole_dir + '_' + str(k) + 'Fold', str(i), 'test')
    _, _, fold_filenames = next(os.walk(fold_path), (None, None, []))
    fold_filenames.sort()
    for file in fold_filenames:
        shutil.copy2(os.path.join(fold_path, file), dst_train)
    # patientNameUnique_fold = list(set(patientName_fold))
    # patientNameUnique_fold.sort()
    patient_pn_fold = [x.split('_')[1] if len(x.split('_')) == 3 else '_'.join(x.split('_')[1:3]) for x in fold_filenames]
    patient_pn_fold = list(set(patient_pn_fold))
    patient_pn_fold.sort()
    filename_patient_pn = [next(filter(lambda filename: name_pn in filename, fold_filenames), None) for name_pn in patient_pn_fold]
    primary_ln = [label_combined[filenames_combined.index(x)] for x in filename_patient_pn]
    fold_num = [i] * len(patient_pn_fold)
    patientName_fold = [x.split('_')[0] for x in patient_pn_fold]
    id_in_fold = range(1,len(patient_pn_fold) + 1)

    with open(log_file_name, 'a') as f:
        s = '-----train fold%d-----' % i
        f.write('%s\n' % s)
        for i in range(0, len(patient_pn_fold)):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(patientName_fold[i], patient_pn_fold[i], fold_num[i], id_in_fold[i], primary_ln[i]))


dst_dev = os.path.join(whole_dir + '_70_10_20', 'dev')
os.makedirs(dst_dev)
for i in dev_folds:
    fold_path = os.path.join(whole_dir + '_' + str(k) + 'Fold', str(i), 'test')
    _, _, fold_filenames = next(os.walk(fold_path), (None, None, []))
    fold_filenames.sort()
    for file in fold_filenames:
        shutil.copy2(os.path.join(fold_path, file), dst_dev)
    patient_pn_fold = [x.split('_')[1] if len(x.split('_')) == 3 else '_'.join(x.split('_')[1:3]) for x in fold_filenames]
    patient_pn_fold = list(set(patient_pn_fold))
    patient_pn_fold.sort()
    filename_patient_pn = [next(filter(lambda filename: name_pn in filename, fold_filenames), None) for name_pn in patient_pn_fold]
    primary_ln = [label_combined[filenames_combined.index(x)] for x in filename_patient_pn]
    fold_num = [i] * len(patient_pn_fold)
    patientName_fold = [x.split('_')[0] for x in patient_pn_fold]
    id_in_fold = range(1,len(patient_pn_fold) + 1)

    with open(log_file_name, 'a') as f:
        s = '-----dev fold%d-----' % i
        f.write('%s\n' % s)
        for i in range(0, len(patient_pn_fold)):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(patientName_fold[i], patient_pn_fold[i], fold_num[i], id_in_fold[i], primary_ln[i]))

dst_test = os.path.join(whole_dir + '_70_10_20', 'test')
os.makedirs(dst_test)
for i in test_folds:
    fold_path = os.path.join(whole_dir + '_' + str(k) + 'Fold', str(i), 'test')
    _, _, fold_filenames = next(os.walk(fold_path), (None, None, []))
    fold_filenames.sort()
    for file in fold_filenames:
        shutil.copy2(os.path.join(fold_path, file), dst_test)
    patient_pn_fold = [x.split('_')[1] if len(x.split('_')) == 3 else '_'.join(x.split('_')[1:3]) for x in fold_filenames]
    patient_pn_fold = list(set(patient_pn_fold))
    patient_pn_fold.sort()
    filename_patient_pn = [next(filter(lambda filename: name_pn in filename, fold_filenames), None) for name_pn in patient_pn_fold]
    primary_ln = [label_combined[filenames_combined.index(x)] for x in filename_patient_pn]
    fold_num = [i] * len(patient_pn_fold)
    patientName_fold = [x.split('_')[0] for x in patient_pn_fold]
    id_in_fold = range(1,len(patient_pn_fold) + 1)

    with open(log_file_name, 'a') as f:
        s = '-----test fold%d-----' % i
        f.write('%s\n' % s)
        for i in range(0, len(patient_pn_fold)):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(patientName_fold[i], patient_pn_fold[i], fold_num[i], id_in_fold[i], primary_ln[i]))

##same 70 10 20 split, 70 used for 5 fold cross validation, then test for the held out 10+20 of data
dst_cv = os.path.join(whole_dir + '_70_10_20', 'cv')
# os.makedirs(dst_cv)
shutil.copytree(dst_train, dst_cv)
dst_cv_test = os.path.join(whole_dir + '_70_10_20', 'cv_test')
shutil.copytree(dst_dev, dst_cv_test)
shutil.copytree(dst_test, dst_cv_test, dirs_exist_ok=True)

#summary of patients, primary and lymph nodes in train, test, dev
log_file_name = os.path.join('/lila/data/deasy/data_harini/wei_head_neck_FDG_FMISO', 'cv_separation_info.txt')
with open(log_file_name, 'w') as file:
    now = time.strftime("%c")
    file.write('==========patient, priamry/LN, id, primary (1)/LN (0) (%s)==========\n' % now)

_, _, fold_filenames = next(os.walk(dst_cv), (None, None, []))
fold_filenames.sort()
patient_pn_fold = [x.split('_')[1] if len(x.split('_')) == 3 else '_'.join(x.split('_')[1:3]) for x in fold_filenames]
patient_pn_fold = list(set(patient_pn_fold))
patient_pn_fold.sort()
filename_patient_pn = [next(filter(lambda filename: name_pn in filename, fold_filenames), None) for name_pn in patient_pn_fold]
primary_ln = [label_combined[filenames_combined.index(x)] for x in filename_patient_pn]
patientName_fold = [x.split('_')[0] for x in patient_pn_fold]
id_in_fold = range(1,len(patient_pn_fold) + 1)

with open(log_file_name, 'a') as f:
    s = '-----test fold%d-----' % i
    f.write('%s\n' % s)
    for i in range(0, len(patient_pn_fold)):
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(patientName_fold[i], patient_pn_fold[i], id_in_fold[i], primary_ln[i]))