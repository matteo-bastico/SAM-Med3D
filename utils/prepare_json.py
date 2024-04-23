import os
import json

base = {
    "name": "TCIA_ct",
    "description": "Dataset from TCIA of healthy patients",
    "labels": {
        "1": "healthy",
    },
    "licence": "yt",
    "modality": {
        "0": "CT"
    },
    "numTest": 0,
    "numTraining": 0,
    "numValidation": 0,
    "reference": "N/A",
    "release": "N/A",
    "tensorImageSize": "3D",
    "training": [],
    "validation": [],
    "testing": []
}

ignore = [
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
splits_path = "../data/medical_preprocessed/healthy/TCIA_ct/Splits"

for file, split in [("Tr.txt", "training"), ("Val.txt", "validation"), ("Ts.txt", "testing")]:
    with open(os.path.join(splits_path, file), 'r') as f:
        # Read the lines of the file into a list
        lines = [line.rstrip('\n') for line in f.readlines()]
    count = 0
    for line in lines:
        if line not in ignore:
            base[split].append(
                {
                    "mask": os.path.join("masks" + file.split('.')[0], line + ".nii.gz"),
                    "image": os.path.join("images" + file.split('.')[0], line + ".nii.gz")
                }
            )
            count += 1
    if split == "training":
        base["numTraining"] = count
    elif split == "validation":
        base["numValidation"] = count
    elif split == "testing":
        base["numTest"] = count

print(base)

with open('../data/medical_preprocessed/healthy/TCIA_ct/dataset.json', 'w') as f:
    json.dump(base, f)