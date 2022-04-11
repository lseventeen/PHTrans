from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/home/lwt/data/synapse/RawData/')
    args = parser.parse_args()

    task_id = 17
    task_name = "AbdominalOrganSegmentation"
    prefix = 'ABD'

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    if isdir(imagestr):
        shutil.rmtree(imagestr)
        shutil.rmtree(imagests)
        shutil.rmtree(labelstr)
        shutil.rmtree(labelsts)

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    val_id = test_id = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]
    img_folder = join(args.dataset_path, "Training/img")
    label_folder = join(args.dataset_path, "Training/label")
    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(img_folder, join=False, suffix='nii.gz')

    for p in train_patients:
        serial_number = int(p[3:7])
        train_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        label_file = join(label_folder, f'label{p[3:]}')
        image_file = join(img_folder, p)
        shutil.copy(image_file, join(
            imagestr, f'{train_patient_name[:7]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

    for p in test_id:
        test_patient = f"img{p:04d}.nii.gz"
        test_patient_name = f'{prefix}_{p:03d}.nii.gz'
        label_file = join(label_folder, f'label{test_patient[3:]}')
        image_file = join(img_folder, test_patient)
        shutil.copy(image_file, join(
            imagests, f'{test_patient_name[:7]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelsts, test_patient_name))
        test_patient_names.append(test_patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "AbdominalOrganSegmentation"
    json_dict['description'] = "Multi-Atlas Labeling Beyond the Cranial Vault Abdominal Organ Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://www.synapse.org/#!Synapse:syn3193805/wiki/217789"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "spleen",
        "02": "right kidney",
        "03": "left kidney",
        "04": "gallbladder",
        "05": "esophagus",
        "06": "liver",
        "07": "stomach",
        "08": "aorta",
        "09": "inferior vena cava",
        "10": "portal vein and splenic vein",
        "11": "pancreas",
        "12": "right adrenal gland",
        "13": "left adrenal gland"}
    )
    json_dict['evaluationClass'] = [8, 4, 3, 2, 6, 11, 1, 7]
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)

    json_dict['test'] = ["./imagesTs/%s" %
                         test_patient_name for test_patient_name in test_patient_names]

    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name,
                              "label": "./labelsTr/%s" % train_patient_name} for train_patient_name in train_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = [i[:7]
                           for i in train_patient_names if int(i[4:7]) not in val_id]
    splits[-1]['val'] = [i[:7]
                         for i in train_patient_names if int(i[4:7]) in val_id]
    save_pickle(splits, join(out_base, "splits_final.pkl"))


if __name__ == "__main__":
   main()

    