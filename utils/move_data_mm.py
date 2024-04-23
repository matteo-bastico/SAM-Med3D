import os.path as osp
import os
import random
import shutil

dataset_root = "../../"
dataset_list = [
    # 'AMOS_val',
    'Bologna_ct',
    # 'TCIA_ct'
]
out_path = "../data/"

"""
    [
        "MPC_123_20061121",
        "MPC_177_20090610",
        "MPC_265_20070102",
        "MPC_381_20060908",
        "MPC_285_20140304",
        "MPC_1568_20070215",
        "MPC_122_20141223",
        "MPC_1402_20171013",
        "MPC_140_20171013",
        "MPC_167_20120210",
        "MPC_2442_20190129",
    ],"""
"""
[
        "PETCT_b79961f3f6",
        "PETCT_80f7760f65",
        "PETCT_5e339b2ecf",
        "PETCT_61d5bc58fc",
        "PETCT_6d62e15c29",
        "PETCT_8311aeddb9",
        "PETCT_544676de40",
        "PETCT_94cc0dac49",
        "PETCT_234f8427c0",
        "PETCT_2716c9bfff",
        "PETCT_7faf36a152",
        "PETCT_2f9aec0275",
        "PETCT_9a66a81ad1",
        "PETCT_ae6a37a9d6",
        "PETCT_ae96f738c0",
        "PETCT_af119148fe",
        "PETCT_bfd89440db",
        "PETCT_c094a24c03",
        "PETCT_cbbc9e2879",
        "PETCT_d626611daf",
        "PETCT_f650c87621",
        "PETCT_f8314eb3f7",
    ]
"""
ignore_patients = [
[
        "MPC_123_20061121",
        "MPC_177_20090610",
        "MPC_265_20070102",
        "MPC_381_20060908",
        "MPC_285_20140304",
        "MPC_1568_20070215",
        "MPC_122_20141223",
        "MPC_1402_20171013",
        "MPC_140_20171013",
        "MPC_167_20120210",
        "MPC_2442_20190129",
        "MPC_136_20150818",
        "MPC_175_20190723",
        "MPC_179_20180507",
        "MPC_249_20130306",

    ]
]

def split_paths(paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    random.shuffle(paths)

    total_paths = len(paths)
    num_train = int(total_paths * train_ratio)
    num_val = int(total_paths * val_ratio)
    num_test = total_paths - num_train - num_val

    train_paths = paths[:num_train]
    val_paths = paths[num_train:num_train + num_val]
    test_paths = paths[num_train + num_val:]

    return train_paths, val_paths, test_paths


for idx, dataset in enumerate(dataset_list):
    ignore = ignore_patients[idx]
    dataset_dir = osp.join(dataset_root, dataset)
    patients = os.listdir(osp.join(dataset_dir, 'PET-CT'))
    print(patients)
    out_dataset_path = osp.join(out_path, dataset)
    os.makedirs(out_dataset_path, exist_ok=True)

    #patients  = list(filter(lambda x: '_' in x, patients))
    # Remove patients with wrong alignment form
    # train_patients, val_patients, test_patients = split_paths(patients)

    # Open the text file in read mode
    with open('../data/Bologna_ct/Splits/Tr.txt', 'r') as file:
        # Read the lines of the file into a list
        train_patients = [line.rstrip('\n') for line in file.readlines()]
    with open('../data/Bologna_ct/Splits/Val.txt', 'r') as file:
        # Read the lines of the file into a list
        val_patients = [line.rstrip('\n') for line in file.readlines()]
    with open('../data/Bologna_ct/Splits/Ts.txt', 'r') as file:
        # Read the lines of the file into a list
        test_patients = [line.rstrip('\n') for line in file.readlines()]

    # Do the splits
    # Shuffle list
    # Take first 70% as train
    # 15% Val / Test
    """
    os.makedirs(osp.dirname(osp.join(out_dataset_path, 'Splits')), exist_ok=True)
    for split, name in zip([train_patients, val_patients, test_patients], ['Tr', 'Val', 'Ts']):
        # Open the file in write mode

        with open(osp.join(out_dataset_path, 'Splits', name + ".txt"), "w") as file:
            # Iterate through the list and write each item to a separate line
            for item in split:
                file.write(str(item) + "\n")
    """
    for patient in train_patients:
        if patient not in ignore:
            print(patient)
            patient_dir = osp.join(dataset_dir, 'PET-CT', patient)
            files = os.listdir(patient_dir)
            out_dir = osp.join(out_dataset_path, 'imagesTr')
            os.makedirs(out_dir, exist_ok=True)
            for filename in files:
                if filename.startswith("CT.") or filename.startswith("CT_"):
                    src_path = osp.join(patient_dir, filename)
                    dest_path = osp.join(out_dir, patient + filename[filename.index('.'):])
                    shutil.copy(src_path, dest_path)

            patient_dir = osp.join(dataset_dir, 'Segmentations', patient)
            files = os.listdir(patient_dir)
            out_dir = osp.join(out_dataset_path, 'masksTr')
            os.makedirs(out_dir, exist_ok=True)
            for filename in files:
                if filename.startswith("mask.") or filename.startswith("mask_"):
                    src_path = osp.join(patient_dir, filename)
                    dest_path = osp.join(out_dir, patient + filename[filename.index('.'):])
                    shutil.copy(src_path, dest_path)

    for patient in val_patients:
        if patient not in ignore:
            print(patient)
            patient_dir = osp.join(dataset_dir, 'PET-CT', patient)
            files = os.listdir(patient_dir)
            out_dir = osp.join(out_dataset_path, 'imagesVal')
            os.makedirs(out_dir, exist_ok=True)
            for filename in files:
                if filename.startswith("CT.") or filename.startswith("CT_"):
                    src_path = osp.join(patient_dir, filename)
                    dest_path = osp.join(out_dir, patient + filename[filename.index('.'):])
                    shutil.copy(src_path, dest_path)

            patient_dir = osp.join(dataset_dir, 'Segmentations', patient)
            files = os.listdir(patient_dir)
            out_dir = osp.join(out_dataset_path, 'masksVal')
            os.makedirs(out_dir, exist_ok=True)
            for filename in files:
                if filename.startswith("mask.") or filename.startswith("mask_"):
                    src_path = osp.join(patient_dir, filename)
                    dest_path = osp.join(out_dir, patient + filename[filename.index('.'):])
                    shutil.copy(src_path, dest_path)

    for patient in test_patients:
        if patient not in ignore:
            print(patient)
            patient_dir = osp.join(dataset_dir, 'PET-CT', patient)
            files = os.listdir(patient_dir)
            out_dir = osp.join(out_dataset_path, 'imagesTs')
            os.makedirs(out_dir, exist_ok=True)
            for filename in files:
                if filename.startswith("CT.") or filename.startswith("CT_"):
                    src_path = osp.join(patient_dir, filename)
                    dest_path = osp.join(out_dir, patient + filename[filename.index('.'):])
                    shutil.copy(src_path, dest_path)

            patient_dir = osp.join(dataset_dir, 'Segmentations', patient)
            files = os.listdir(patient_dir)
            out_dir = osp.join(out_dataset_path, 'masksTs')
            os.makedirs(out_dir, exist_ok=True)
            for filename in files:
                if filename.startswith("mask.") or filename.startswith("mask_"):
                    src_path = osp.join(patient_dir, filename)
                    dest_path = osp.join(out_dir, patient + filename[filename.index('.'):])
                    shutil.copy(src_path, dest_path)


    # Copy CT and bone_filled to correct folder ->
    # Put in the correct output Split

