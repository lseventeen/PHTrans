from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunet.paths import nnUNet_raw_data
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/home/lwt/data/ACDC/training')
    args = parser.parse_args()

    task_id = 27
    task_name = "ACDC"
    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_folder = join(nnUNet_raw_data, foldername)
    test_id = [2, 3, 8, 9, 12, 14, 17, 24, 42, 48,
               49, 53, 55, 64, 67, 79, 81, 88, 92, 95]
    val_id = [89, 90, 91, 93, 94, 96, 97, 98, 99, 100]
    imagestr = join(out_folder, "imagesTr")
    labelstr = join(out_folder, "labelsTr")
    imagests = join(out_folder, "imagesTs")
    labelsts = join(out_folder, "labelsTs")
    if isdir(imagestr):
        shutil.rmtree(imagestr)
        shutil.rmtree(labelstr)
        shutil.rmtree(imagests)
        shutil.rmtree(labelsts)

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    # train
    all_train_files = []
    all_test_files = []
    patient_dirs = subfolders(args.dataset_path, prefix="patient")
    for p in patient_dirs:
        current_dir = p
        data_files = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find(
            "_gt") == -1 and i.find("_4d") == -1]
        corresponding_seg_files = [i[:-7] + "_gt.nii.gz" for i in data_files]
        for d, s in zip(data_files, corresponding_seg_files):
            patient_identifier = d.split("/")[-1][:-7]
            id = int(patient_identifier[7:10])
            if id in test_id:
                all_test_files.append(patient_identifier + "_0000.nii.gz")
                shutil.copy(d, join(out_folder, "imagesTs",
                            patient_identifier + "_0000.nii.gz"))
                shutil.copy(s, join(out_folder, "labelsTs",
                            patient_identifier + ".nii.gz"))
            else:
                all_train_files.append(patient_identifier + "_0000.nii.gz")
                shutil.copy(d, join(out_folder, "imagesTr",
                            patient_identifier + "_0000.nii.gz"))
                shutil.copy(s, join(out_folder, "labelsTr",
                            patient_identifier + ".nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "ACDC"
    json_dict['description'] = "cardias cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see ACDC challenge"
    json_dict['licence'] = "see ACDC challenge"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "RV",
        "2": "MLV",
        "3": "LVC"
    }
    json_dict['evaluationClass'] = [1, 2, 3]
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = len(all_test_files)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-12]} for i in
                             all_train_files]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" %
                         i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

    # create a dummy split (patients need to be separated)
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = [i[:-12]
                           for i in all_train_files if int(i[7:10]) not in val_id]
    splits[-1]['val'] = [i[:-12]
                         for i in all_train_files if int(i[7:10]) in val_id]
    save_pickle(splits, join(out_folder, "splits_final.pkl"))

if __name__ == "__main__":
    main()
    