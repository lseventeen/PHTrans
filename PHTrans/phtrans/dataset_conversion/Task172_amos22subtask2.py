from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/home/lwt/data/amos22')
    args = parser.parse_args()

    task_id = 172
    task_name = "AMOS22st2"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    if isdir(imagestr):
        shutil.rmtree(imagestr)
        shutil.rmtree(imagests)
        shutil.rmtree(labelstr)

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_image_folder = join(args.dataset_path, "imagesTr")
    train_label_folder = join(args.dataset_path, "labelsTr")
    test_image_folder = join(args.dataset_path, "imagesTs")
    dataset_info = load_json(join(args.dataset_path, "task2_dataset.json"))
    train_info = dataset_info['training']
    test_info = dataset_info['test']
    train_ids = []
    test_ids = []
    for i in train_info:
        id = i["image"].split("/")[-1]
        train_ids.append(id)
        shutil.copy(join(train_image_folder, id), join(imagestr, id.split(".")[0]+"_0000.nii.gz"))
        shutil.copy(join(train_label_folder, id), join(labelstr, id))
    for i in test_info:
        id = i.split("/")[-1]
        test_ids.append(id)
        shutil.copy(join(test_image_folder, id), join(imagests, id.split(".")[0]+"_0000.nii.gz"))
        
    json_dict = OrderedDict()
    json_dict['name'] = "AMOS"
    json_dict['description'] = "MICCAI2022 Multi-Modality Abdominal Multi-Organ Segmentation Task 2"
    json_dict['author'] = "Yuanfeng Ji"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "SRIDB x CUHKSZ x HKU x SYSU x LGCHSZ x LGPHSZ"
    json_dict['licence'] =  "CC-BY-SA 4.0"
    json_dict['release'] = "1.0 01/05/2022"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "0": "background",
        "1": "spleen", 
        "2": "right kidney", 
        "3": "left kidney", 
        "4": "gall bladder", 
        "5": "esophagus", 
        "6": "liver", 
        "7": "stomach", 
        "8": "arota", 
        "9": "postcava", 
        "10": "pancreas", 
        "11": "right adrenal gland", 
        "12": "left adrenal gland", 
        "13": "duodenum", 
        "14": "bladder", 
        "15": "prostate/uterus"
        })
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)


    json_dict['training'] = dataset_info['training']
    json_dict['test'] = dataset_info['test']
    save_json(json_dict, os.path.join(out_base, "dataset.json"))

if __name__ == "__main__":
   main()