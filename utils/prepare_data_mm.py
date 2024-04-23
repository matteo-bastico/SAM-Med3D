# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet.py
@Time    :   2023/12/10 23:07:39
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   pre-process nnUNet-style dataset into SAM-Med3D-style
'''

import os.path as osp
import os
import json
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm
import torchio as tio

def resample_nii(input_path: str, output_path: str, target_spacing: tuple = (1.5, 1.5, 1.5), n=None, reference_image=None, mode="linear"):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    
    # Load the nii.gz file using torchio
    subject = tio.Subject(
        img=tio.ScalarImage(input_path)
    )
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)
    resampled_subject = resampler(subject)
    
    if(n!=None):
        image = resampled_subject.img
        tensor_data = image.data
        if(isinstance(n, int)):
            n = [n]
        for ni in n:
            tensor_data[tensor_data == ni] = -1
        tensor_data[tensor_data != -1] = 0
        tensor_data[tensor_data != 0] = 1
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        reference_size = reference_image.shape[1:]  # omitting the channel dimension
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img
    
    save_image.save(output_path)

dataset_root = "../data"
dataset_list = [
    # 'AMOS_val',
    'Bologna_ct',
    'TCIA_ct'
]

target_dir = "../data/medical_preprocessed"


for dataset in dataset_list:
    dataset_dir = osp.join(dataset_root, dataset)
    meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))

    print(meta_info['name'], meta_info['modality'])
    num_classes = len(meta_info["labels"])-1
    print("num_classes:", num_classes, meta_info["labels"])

    for split in ['Tr', 'Val', 'Ts']:
        resample_dir = osp.join(dataset_dir, "images" + split + "_1.5")
        os.makedirs(resample_dir, exist_ok=True)
        for idx, cls_name in meta_info["labels"].items():
            cls_name = cls_name.replace(" ", "_")
            idx = int(idx)
            # dataset_name = dataset.split("_", maxsplit=1)[1]
            dataset_name = dataset
            target_cls_dir = osp.join(target_dir, cls_name, dataset_name)
            target_img_dir = osp.join(target_cls_dir, "images" + split)
            target_mask_dir = osp.join(target_cls_dir, "masks" + split)
            os.makedirs(target_img_dir, exist_ok=True)
            os.makedirs(target_mask_dir, exist_ok=True)

            if split == 'Tr':
                info = "training"
            elif split == 'Val':
                info = "validation"
            elif split == 'Ts':
                info = "testing"

            for item in tqdm(meta_info[info], desc=f"{dataset_name}-{cls_name}"):
                img, mask = item["image"], item["mask"]
                img = osp.join(dataset_dir, img)
                mask = osp.join(dataset_dir, mask)
                if img.endswith('.nii'):
                    resample_img = osp.join(resample_dir, osp.basename(img + '.gz'))
                else:
                    resample_img = osp.join(resample_dir, img)

                if(not osp.exists(resample_img)):
                    resample_nii(img, resample_img)
                img = resample_img

                target_img_path = osp.join(target_img_dir, osp.basename(img))
                target_mask_path = osp.join(target_mask_dir, osp.basename(mask))
                mask_img = nib.load(mask)
                spacing = tuple(mask_img.header['pixdim'][1:4])
                spacing_voxel = spacing[0] * spacing[1] * spacing[2]
                mask_arr = mask_img.get_fdata()
                # Here we round to int because it gives some problem with some datasets
                mask_arr = np.around(mask_arr)
                mask_arr[mask_arr != idx] = 0
                mask_arr[mask_arr != 0] = 1
                volume = mask_arr.sum()*spacing_voxel
                if(volume<10):
                    print("skip", target_img_path)
                    continue

                reference_image = tio.ScalarImage(img)
                if(meta_info['name']=="kits23" and idx==1):
                    resample_nii(mask, target_mask_path, n=[1,2,3], reference_image=reference_image, mode="nearest")
                else:
                    resample_nii(mask, target_mask_path, n=idx, reference_image=reference_image, mode="nearest")
                shutil.copy(img, target_img_path)



