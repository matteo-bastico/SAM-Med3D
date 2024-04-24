from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
from torchio.data.io import sitk_to_nib
import torch
import numpy as np
import pandas as pd
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator


class Dataset_Union_ALL(Dataset): 
    def __init__(
            self,
            paths,
            ann_index=['ID', 'MDS'],
            annotations=["BM DS", "FL", "PM", "EM"],
            classes=["healthy", "mm"],
            mode='train',
            data_type='Tr',
            image_size=128,
            transform=None,
            threshold=500,
            split_num=1,
            split_idx=0,
            pcc=False,
            get_all_meta_info=False
    ):
        self.paths = paths
        self.ann_index = ann_index
        self.annotations = annotations
        self.classes = classes
        self.data_type = data_type
        self.split_num = split_num
        self.split_idx = split_idx

        self._set_file_paths(self.paths)
        self._set_annotations()
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
        self.mode = mode
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotations = self.annotations.iloc[index]
        path_index = self.image_paths.index(annotations.name)
        sitk_image = sitk.ReadImage(self.image_paths[path_index])
        sitk_mask = sitk.ReadImage(self.mask_paths[path_index])
        # annotations = self.annotations.loc[[self.image_paths[index]]]
        # Retrive class from path
        for item in self.classes:
            if item in self.image_paths[path_index]:
                cls = self.classes.index(item)
                break
        annotations = torch.cat([torch.tensor([cls]), torch.tensor(annotations.values.squeeze())])

        if sitk_image.GetOrigin() != sitk_mask.GetOrigin():
            sitk_image.SetOrigin(sitk_mask.GetOrigin())
        if sitk_image.GetDirection() != sitk_mask.GetDirection():
            sitk_image.SetDirection(sitk_mask.GetDirection())

        sitk_image_arr, _ = sitk_to_nib(sitk_image)
        sitk_mask_arr, _ = sitk_to_nib(sitk_mask)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk_image_arr),
            mask=tio.LabelMap(tensor=sitk_mask_arr),
        )

        if '_ct/' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])

        if(self.pcc):
            print("using pcc setting")
            # crop from random click point
            random_index = torch.argwhere(subject.mask.data == 1)
            if(len(random_index)>=1):
                random_index = random_index[np.random.randint(0, len(random_index))]
                # print(random_index)
                crop_mask = torch.zeros_like(subject.mask.data)
                # print(crop_mask.shape)
                crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
                subject.add_image(tio.LabelMap(tensor=crop_mask,
                                                affine=subject.mask.affine),
                                    image_name="crop_mask")
                subject = tio.CropOrPad(mask_name='crop_mask', 
                                        target_shape=(self.image_size,self.image_size,self.image_size))(subject)

        # Activate plot to check if we are doing things good
        # subject.plot()

        if subject.mask.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        
        if self.mode == "train" and self.data_type == 'Tr':
            return subject.image.data.clone().detach().float(), subject.mask.data.clone().detach().float(), annotations.float()
        elif self.get_all_meta_info:
            meta_info = {
                "image_path": self.image_paths[index],
                "origin": sitk_mask.GetOrigin(),
                "direction": sitk_mask.GetDirection(),
                "spacing": sitk_mask.GetSpacing(),
            }
            return subject.image.data.clone().detach().float(), subject.mask.data.clone().detach().float(), annotations, meta_info
        else:
            return subject.image.data.clone().detach().float(), subject.mask.data.clone().detach().float(), annotations, self.image_paths[index]
 
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.mask_paths = []

        # if ${path}/masksTr exists, search all .nii.gz
        for path in paths:
            d = os.path.join(path, f'masks{self.data_type}')
            if os.path.exists(d):
                for name in os.listdir(d):
                    base = os.path.basename(name).split('.nii.gz')[0]
                    mask_path = os.path.join(path, f'masks{self.data_type}', f'{base}.nii.gz')
                    self.image_paths.append(mask_path.replace('masks', 'images'))
                    self.mask_paths.append(mask_path)

    def _set_annotations(self):
        annotations_dfs = []
        for idx, path in enumerate(self.paths):
            d = os.path.join(path, f'annotations{self.data_type}.csv')
            if os.path.exists(d):
                # Loaad from csv
                df = pd.read_csv(d)
                df.set_index(self.ann_index[idx], inplace=True)
                new_indices = [os.path.join(path, f'images{self.data_type}', f'{index}.nii.gz') for index in df.index]
                df.index = new_indices
                # Filter only column in annotations list
                df = df[self.annotations]
                annotations_dfs.append(df)

        self.annotations = pd.concat(annotations_dfs)
        # self.annotations = concatenation of self.annotations_dfs


class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            for dt in ["Tr", "Val", "Ts"]:
                d = os.path.join(path, f'labels{dt}')
                if os.path.exists(d):
                    for name in os.listdir(d):
                        base = os.path.basename(name).split('.nii.gz')[0]
                        label_path = os.path.join(path, f'labels{dt}', f'{base}.nii.gz') 
                        self.image_paths.append(label_path.replace('labels', 'images'))
                        self.label_paths.append(label_path)
        self.image_paths = self.image_paths[self.split_idx::self.split_num]
        self.label_paths = self.label_paths[self.split_idx::self.split_num]


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset): 
    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])


        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        

        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
    
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace('images', 'labels'))


if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL(
        paths=['../data/medical_preprocessed/healthy/TCIA_ct',
               '../data/medical_preprocessed/mm/Bologna_ct'],
        data_type='Tr',
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='mask', target_shape=(256, 128, 512)),
        ]),
        ann_index=['MPC', 'MPC'],
        annotations=["BM DS", "FL (sec Impetus)", "PM ", "EM "],
        threshold=0,
        pcc=False
    )

    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=4,
        shuffle=True
    )
    for i,j,a in test_dataloader:
        print(a)
        # print(i.shape)
        # print(j.shape)
        # print(n)
        continue

