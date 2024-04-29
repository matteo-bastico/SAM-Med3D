# set up environment
import numpy as np
import random 
import datetime
import logging
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import math
import wandb
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas
from utils.loss import MultipleLoss, FLWithLogitsLoss
from torch.nn import BCEWithLogitsLoss, MSELoss
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

try:
    import idr_torch
except ImportError:
    pass

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='multitask_freeze_e100_lr8e-4_linear_bs4_as8')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_mlp')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--img_size', type=int, nargs='+', default=[128])

args = parser.parse_args()

if len(args.img_size) == 1 or isinstance(args.img_size, int):
    args.img_size = (args.img_size[0], args.img_size[0], args.img_size[0])
elif len(args.img_size) != 3:
    raise ValueError("img_size should be either a single value or a list of 3 values")

device = args.device
LOG_OUT_DIR = join(args.work_dir, args.task_name)
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    print(sam_model)
    return sam_model


def get_dataloader(args):
    test_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(target_shape=(args.img_size[0], args.img_size[1], args.img_size[2])),
        ]),  # or maybe crop always in the center for val
        threshold=0,
        mode="test",
        data_type="Test",
        ann_index=['MPC', 'MPC'],
        annotations=["BM DS", "FL (sec Impetus)", "PM ", "EM "],
        classes=["healthy", "mm"],
    )
    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return test_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, args, run=None):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args

        # Define which metrics compute for each task
        self.metrics_fn = [
            [self.get_accuracy, self.get_precision, self.get_recall, self.get_f1],  # self.get_roc_auc
            [self.get_mae],
            [self.get_mae],
            [self.get_accuracy, self.get_precision, self.get_recall, self.get_f1],  # self.get_roc_auc
            [self.get_accuracy, self.get_precision, self.get_recall, self.get_f1],  # self.get_roc_auc
        ]

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        # Wandb log
        self.run = run

    def get_accuracy(self, input, target):
        total_samples = len(input)
        probabilities = torch.sigmoid(input)
        binary_predictions = torch.round(probabilities)
        correct_predictions = torch.sum(binary_predictions == target)
        return (correct_predictions / total_samples).item()

    def get_precision(self, input, target):
        probabilities = torch.sigmoid(input)
        binary_predictions = torch.round(probabilities)
        true_positives = torch.sum(binary_predictions * target)
        predicted_positives = torch.sum(binary_predictions)
        return (true_positives / (predicted_positives + 1e-7)).item()

    def get_recall(self, input, target):
        probabilities = torch.sigmoid(input)
        binary_predictions = torch.round(probabilities)
        true_positives = torch.sum(binary_predictions * target)
        actual_positives = torch.sum(target)
        return (true_positives / (actual_positives + 1e-7)).item()

    def get_f1(self, input, target):
        prec = self.get_precision(input, target)
        rec = self.get_recall(input, target)
        return 2 * ((prec * rec) / (prec + rec + 1e-7))

    def get_roc_auc(self, input, target):
        target_np = target.cpu().detach().numpy()
        input_np = input.detach().cpu().numpy()
        if len(np.unique(target_np)) == 1:
            return np.nan
        else:
            return roc_auc_score(target_np, input_np)

    def get_mae(self, input, target):
        abs_error = torch.abs(input - target)
        return torch.mean(abs_error).item()

    def batch_forward(
            self,
            sam_model,
            image_embedding,
            mask3D,
            gt,
            points=None
    ):
        low_res_masks = F.interpolate(
            mask3D.float(),
            size=(args.img_size[0] // 4, args.img_size[1] // 4, args.img_size[2] // 4),
            mode="nearest"  # The mask should be binary
        )
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, N_points, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        )

        return predictions

    def get_metrics(self, predictions, gt):
        out = dict()
        for i, metrics in enumerate(self.metrics_fn):
            out["Task " + str(i)] = dict()
            input = predictions[i]
            target = gt[i]
            for metric in metrics:
                out["Task " + str(i)][metric.__name__.replace('get_', '')] = metric(input, target)
        return out

    def eval_epoch(self):
        with torch.no_grad():

            self.model.eval()
            sam_model = self.model

            tbar = tqdm(self.dataloaders)

            all_predictions = []
            all_gt = []

            for step, (image3D, mask3D, gt, path) in enumerate(tbar):

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                mask3D = mask3D.to(device).type(torch.long)
                gt = gt.to(device).type(torch.float)

                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)

                    predictions = self.batch_forward(
                        sam_model,
                        image_embedding,
                        mask3D,
                        gt,
                    )

                all_predictions.append(predictions)
                all_gt.append(gt)

            all_gt = torch.stack(all_gt).squeeze()
            # Transpose the list of lists
            transposed_lists = list(map(list, zip(*all_predictions)))
            # List to hold stacked tensors
            stacked_tensors = []
            # Iterate over each transposed list of tensors
            for tensor_list in transposed_lists:
                # Stack tensors along a new dimension (default is dim=0)
                stacked_tensor = torch.stack(tensor_list).squeeze(-1)
                stacked_tensors.append(stacked_tensor)
            all_predictions = stacked_tensors

            epoch_metrics = self.get_metrics(all_predictions, [all_gt[:, i:i + 1] for i in range(all_gt.size(1))])

            return epoch_metrics, all_gt.cpu().numpy(), torch.stack(all_predictions).cpu().numpy()

    def test(self):
        return self.eval_epoch()


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


if __name__ == "__main__":
    test_dataset = Dataset_Union_ALL(
        paths=['data/medical_preprocessed/healthy/TCIA_ct',
               'data/medical_preprocessed/mm/Bologna_ct'],
        data_type='Ts',
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(target_shape=(args.img_size[0], args.img_size[1], args.img_size[2])),
        ]),
        ann_index=['MPC', 'MPC'],
        annotations=["BM DS", "FL (sec Impetus)", "PM ", "EM "],
        threshold=0,
        pcc=False
    )
    test_dataloader = Union_Dataloader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # dataloaders = get_dataloaders(args)
    # Build model
    model = build_model(args)
    # TODO: if it is trained on DDP we have to take module
    state_dict = torch.load(args.checkpoint, map_location=device)
    model_state_dict = state_dict["model_state_dict"]

    model.load_state_dict(
        model_state_dict,
        strict=True
    )
    # Create trainer
    trainer = BaseTrainer(model, test_dataloader, args)
    # Train
    metrics, true_labels, predictions = trainer.test()

    confusion_matrices = []
    for i in range(true_labels.shape[1]):
        # Binary
        if len(np.unique(true_labels[:, i])) == 2:
            probabilities = sigmoid(predictions[i])
            binary_predictions = np.round(probabilities)
            confusion_matrices.append(confusion_matrix(true_labels[:, i], binary_predictions))
        # Regression
        else:
            rounded_predictions = np.round(predictions[i])
            confusion_matrices.append(confusion_matrix(true_labels[:, i], rounded_predictions))

    for i, confusion_matrix in enumerate(confusion_matrices):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
        )
        disp.plot()
        disp.ax_.set_title("Confusion Matrix Task " + str(i))
        plt.show()

    # AUC-ROC
    for i in range(true_labels.shape[1]):
        probabilities = sigmoid(predictions[i])
        # Binary
        if len(np.unique(true_labels[:, i])) == 2:
            probabilities = sigmoid(predictions[i])
            fpr, tpr, _ = roc_curve(true_labels[:, i], probabilities)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            lw = 2
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=lw,
                label="ROC curve (area = %0.2f)" % roc_auc
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve Task " + str(i))
            plt.legend(loc="lower right")
            plt.show()


