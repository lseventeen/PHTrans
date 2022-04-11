import glob
import os
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from medpy import metric
from nnunet.paths import nnUNet_raw_data
import pandas as pd


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def get_dice_hd95(pre_path, experiment_id, task_id):
    task = convert_id_to_task_name(task_id)
    label_path = join(nnUNet_raw_data, task)
    predict_list = sorted(glob.glob(os.path.join(pre_path, '*nii.gz')))
    label_list = sorted(
        glob.glob(os.path.join(label_path, 'labelsTs', '*nii.gz')))

    print("loading success...")
    dataset = load_json(join(label_path, 'dataset.json'))
    metric_list = 0.0
    for predict, label in zip(predict_list, label_list):
        case = predict.split('/')[-1]
        print(case)
        predict, label, = read_nii(predict), read_nii(label)
        metric_i = []
        for i in dataset['evaluationClass']:
            metric_i.append(calculate_metric_percase(predict == i, label == i))
        metric_list += np.array(metric_i)
        print('case %s mean_dice %f mean_hd95 %f' %
              (case, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(predict_list)
    clm = ["DSC", "HD", "Aotra", "Gallbladder", "Kidnery(L)", "Kidnery(R)", "Liver", "Pancreas",
           "Spleen", "Stomach"] if task_id == 17 else ["DSC", "HD", "RV", "MLV", "LVC"]
    for i in range(len(dataset['evaluationClass'])):
        print('Mean class %s mean_dice %f mean_hd95 %f' %
              (clm[i+2], metric_list[i][0], metric_list[i][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (
        performance, mean_hd95))

    idx = experiment_id
    data = np.hstack((performance*100, mean_hd95,
                     metric_list[:, 0]*100)).reshape(1, len(dataset['evaluationClass'])+2)
    df = pd.DataFrame(data, index=[idx], columns=clm)
    df.to_csv(join(pre_path, f"{experiment_id}_result.cvs"))

    df.to_excel(
        join(pre_path, f"{experiment_id}_result.xlsx"), sheet_name='Synapse')


if __name__ == '__main__':
    pre_path = "/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/PHTransTrainer__nnUNetPlansv2.1/fold_0/phtrans_220410_211502/validation_raw_postprocessed"
    experiment_id = "phtrans_220410_211502"
    get_dice_hd95(pre_path, experiment_id, 17)
