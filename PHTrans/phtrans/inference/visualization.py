import numpy as np
import os
import glob
import SimpleITK as sitk
from scipy import ndimage
import cv2
from PIL import Image
import matplotlib.pyplot as plt  # 载入需要的库


def set_colour(img_path,data_path,slice,name):
    img_nii = sitk.ReadImage(img_path)
    pre_nii = sitk.ReadImage(data_path) # 读取其中一个volume数据

    img_3D = sitk.GetArrayFromImage(img_nii) # 提取数据中心的array
    pre_3D = sitk.GetArrayFromImage(pre_nii) # 提取数据中心的array

    img = img_3D[slice,...]
    pre = pre_3D[slice,...]

    H,W = img.shape
    colour = np.zeros((H,W,3))

    for i in range(H):
        for j in range(W):
            if pre[i,j] == 1:
                colour[i,j,:] = [0,0,255] # 纯红
            elif pre[i,j] == 2:
                colour[i,j,:] = [255,255,0] #青色
            elif pre[i,j] == 3:
                colour[i,j,:] = [0,128,0] #纯绿
                
            elif pre[i,j] == 4:
                colour[i,j,:] = [203,255,192] #粉红
            elif pre[i,j] == 6:
                colour[i,j,:] = [255,0,0] #蓝色
            elif pre[i,j] == 7:
                colour[i,j,:] = [255,0,255] #纯黄
            elif pre[i,j] == 8:
                colour[i,j,:] = [128,0,128] #紫色
            elif pre[i,j] == 11:
                colour[i,j,:] = [218,185,255] #桃色
            else:
                colour[i,j,:] = [img[i,j],img[i,j],img[i,j]]
    # colour = Image.fromarray(colour.astype('uint8')).convert('RGB')
    # colour.save(f"nnunet/inference/save_picture/{name}.png")
    cv2.imwrite(f"nnunet/inference/save_picture/{name}.png",colour)


if __name__ == '__main__':
    img="/home/lwt/data/synapse/RawData/Training/img/img0008.nii"
    phtrans="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/PFTC_dpR0.2_dpC_0.1_220224_114831/model_best/ABD_008.nii"
    nnformer="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/nnformer_pretrain_211130_190450/model_best/ABD_008.nii"
    cotr="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/cotr_220102_220809/model_best/ABD_008.nii"
    nnunet="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/nnUNet_220127_001435/model_best/ABD_008.nii"
    gt = "/home/lwt/data_pro/nnUNet_preprocessed/Task017_AbdominalOrganSegmentation/gt_segmentations/ABD_008.nii"

    slice = 45
    set_colour(img,phtrans,slice,"phtrans2")
    set_colour(img,gt,slice,"gt2")
    set_colour(img,nnformer,slice,"nnformer2")
    set_colour(img,cotr,slice," cotr2")
    set_colour(img,nnunet,slice," nnunet2")