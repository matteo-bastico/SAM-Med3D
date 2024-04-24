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
from utils.loss import MultipleLoss
from torch.nn import BCEWithLogitsLoss, MSELoss
from sklearn.metrics import roc_auc_score

try:
    import idr_torch
except ImportError:
    pass

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_mlp')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)
parser.add_argument('--freeze_encoder', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, nargs='+', default=128)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

# losses
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1, 1, 1, 1, 1])

# wandb
parser.add_argument('--log_all', action='store_true', default=False)
parser.add_argument('--entity', type=str, default='phd-matteo')
parser.add_argument('--project', type=str, default='DIMA-MM')
parser.add_argument('--wandb_mode', type=str, default='online')

args = parser.parse_args()

if len(args.img_size) == 1 or isinstance(args.img_size, int):
    args.img_size = (args.img_size[0], args.img_size[0], args.img_size[0])
elif len(args.img_size) != 3:
    raise ValueError("img_size should be either a single value or a list of 3 values")

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def sum_metrics_dicts(dict1, dict2):
    result_dict = {}
    for key in dict1:
        if isinstance(dict1[key], dict):
            result_dict[key] = sum_metrics_dicts(dict1[key], dict2[key])
        else:
            result_dict[key] = dict1[key] + dict2[key]
    return result_dict


def divide_metrics(dictionary, divisor):
    result_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result_dict[key] = divide_metrics(value, divisor)
        else:
            result_dict[key] = value / divisor
    return result_dict


def flatten_metrics(nested_dict, prefix='', separator='/', suffix=''):
    flattened_dict = {}
    for key1, inner_dict in nested_dict.items():
        for key2, value in inner_dict.items():
            flattened_key = f"{prefix}{key1}{separator}{key2}{suffix}"
            flattened_dict[flattened_key] = value
    return flattened_dict


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    print(sam_model)
    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name='mask', target_shape=(args.img_size[0],args.img_size[1],args.img_size[2])),  # crop only object region
            tio.RandomFlip(axes=(0, 1, 2)),
        ]),
        threshold=0,
        ann_index=['MPC', 'MPC'],
        annotations=["BM DS", "FL (sec Impetus)", "PM ", "EM "],
        classes=["healthy", "mm"],
    )
    val_dataset = Dataset_Union_ALL(
        paths=img_datas,
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(target_shape=(args.img_size[0],args.img_size[1],args.img_size[2])),  # crop always in the center for val
        ]),
        threshold=0,
        mode="validation",
        data_type="Val",
        ann_index=['MPC', 'MPC'],
        annotations=["BM DS", "FL (sec Impetus)", "PM ", "EM "],
        classes=["healthy", "mm"],
    )

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = Union_Dataloader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_dataloader, val_dataloader


class BaseTrainer:
    def __init__(self, model, dataloaders, args, run=None):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.step_best_loss = np.inf
        self.losses = []
        self.val_losses = []
        self.single_losses = []
        self.val_single_losses = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        # Define which metrics compute for each task
        self.metrics_fn = [
            [self.get_accuracy, self.get_precision, self.get_recall, self.get_f1],  # self.get_roc_auc
            [self.get_mae],
            [self.get_mae],
            [self.get_accuracy, self.get_precision, self.get_recall, self.get_f1],  # self.get_roc_auc
            [self.get_accuracy, self.get_precision, self.get_recall, self.get_f1],  # self.get_roc_auc
        ]
        self.metrics = {"Task {}".format(i): {func.__name__.replace("get_", ""): [] for func in sublist}
                            for i, sublist in enumerate(self.metrics_fn)}
        self.val_metrics = {"Task {}".format(i): {func.__name__.replace("get_", ""): [] for func in sublist}
                            for i, sublist in enumerate(self.metrics_fn)}
        self.best_metrics = {"Task {}".format(i): {func.__name__.replace("get_", ""): 0 for func in sublist}
                            for i, sublist in enumerate(self.metrics_fn)}
        self.step_best_metrics = {"Task {}".format(i): {func.__name__.replace("get_", ""): 0 for func in sublist}
                            for i, sublist in enumerate(self.metrics_fn)}

        if(args.resume):
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

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
        return (true_positives / (predicted_positives + 1e-10)).item()

    def get_recall(self, input, target):
        probabilities = torch.sigmoid(input)
        binary_predictions = torch.round(probabilities)
        true_positives = torch.sum(binary_predictions * target)
        actual_positives = torch.sum(target)
        return (true_positives / (actual_positives + 1e-10)).item()

    def get_f1(self, input, target):
        prec = self.get_precision(input, target)
        rec = self.get_recall(input, target)
        return 2 * ((prec * rec) / (prec + rec + 1e-10))

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

    def set_loss_fn(self):
        self.loss_fn = MultipleLoss(
            loss_fns=[
                BCEWithLogitsLoss(),
                MSELoss(),
                MSELoss(),
                BCEWithLogitsLoss(),
                BCEWithLogitsLoss()
            ],
            reduction='none',
        )
        self.loss_weights = self.args.loss_weights
    
    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        self.optimizer = torch.optim.AdamW([
            {'params': sam_model.image_encoder.parameters()}, # , 'lr': self.args.lr * 0.1},
            {'params': sam_model.prompt_encoder.parameters() , 'lr': self.args.lr * 0.1},
            {'params': sam_model.mask_decoder.parameters(), 'lr': self.args.lr * 0.1},
        ], lr=self.args.lr, betas=(0.9,0.999), weight_decay=self.args.weight_decay)

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if(self.args.allow_partial_weight):
                # Remove wrongly shaped parameters
                def filter_wrong_shape(state_dict, model):
                    filtered_state_dict = {}
                    for name, param in state_dict.items():
                        if name in model.state_dict():
                            if param.shape == model.state_dict()[name].shape:
                                filtered_state_dict[name] = param
                            else:
                                print(
                                    f"Ignoring parameter '{name}' due to wrong shape: {param.shape}. Expected shape: {model.state_dict()[name].shape}")
                        else:
                            print(f"Ignoring parameter '{name}' because it does not exist in the model.")
                    return filtered_state_dict

                if self.args.multi_gpu:
                    filtered_state_dict = filter_wrong_shape(last_ckpt['model_state_dict'], self.model.module)
                    self.model.module.load_state_dict(filtered_state_dict, strict=False)
                else:
                    filtered_state_dict = filter_wrong_shape(last_ckpt['model_state_dict'], self.model)
                    self.model.load_state_dict(filtered_state_dict, strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0 
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.metrics = last_ckpt['metrics']
                self.best_loss = last_ckpt['best_loss']
                self.best_metrics = last_ckpt['best_metrics']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "metrics": self.metrics,
            "best_loss": self.best_loss,
            "best_metrics": self.best_metrics,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

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
        # Compute loss
        losses = self.loss_fn(predictions, [gt[:, i:i+1] for i in range(gt.size(1))])

        return predictions, losses

    def get_metrics(self, predictions, gt):
        out = dict()
        for i, metrics in enumerate(self.metrics_fn):
            out["Task " + str(i)] = dict()
            input = predictions[i]
            target = gt[i]
            for metric in metrics:
                out["Task " + str(i)][metric.__name__.replace('get_', '')] = metric(input, target)
        return out

    def append_metrics(self, metrics):
        for key in self.metrics:
            for sub_key in self.metrics[key]:
                self.metrics[key][sub_key].append(metrics[key][sub_key])

    def append_val_metrics(self, metrics):
        for key in self.val_metrics:
            for sub_key in self.val_metrics[key]:
                self.val_metrics[key][sub_key].append(metrics[key][sub_key])

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_losses = []
        epoch_metrics = None
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders[0])
        else:
            tbar = self.dataloaders[0]

        self.optimizer.zero_grad()
        step_loss = 0
        step_losses = []
        step_metrics = None

        for step, (image3D, mask3D, gt) in enumerate(tbar):
            # For a sample GT is
            # [CLS, Bone Marrow (BM DS), Focal (FL), Paramedullary (PM), Extramedullary (EM)]
            my_context = self.model.no_sync if self.args.rank != -1 and (step + 1) % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)
                
                image3D = image3D.to(device)
                mask3D = mask3D.to(device).type(torch.long)
                gt = gt.to(device).type(torch.float)

                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    predictions, losses = self.batch_forward(
                        sam_model,
                        image_embedding,
                        mask3D,
                        gt,
                    )
                    loss = sum([weight * loss for weight, loss in zip(self.loss_weights, losses)])

                cur_losses = losses
                metrics = self.get_metrics(predictions, [gt[:, i:i+1] for i in range(gt.size(1))])
                cur_metrics = metrics
                if epoch_metrics is None:
                    epoch_metrics = cur_metrics
                else:
                    epoch_metrics = sum_metrics_dicts(epoch_metrics, cur_metrics)

                epoch_loss += loss.item()
                epoch_losses.append(losses)
                cur_loss = loss.item()
                loss /= self.args.accumulation_steps
                
                self.scaler.scale(loss).backward()    

            if (step + 1) % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                step_loss += cur_loss  # Added this here otherwise the step_loss is not counting for the last cur_loss
                step_losses.append(cur_losses)
                if step_metrics is None:
                    step_metrics = cur_metrics
                else:
                    step_metrics = sum_metrics_dicts(step_metrics, cur_metrics)

                print_loss = step_loss / self.args.accumulation_steps
                print_losses = torch.tensor(step_losses).detach().cpu().numpy()
                print_losses = np.mean(print_losses, axis=0)

                # This metric considers all the batches accumulated
                print_metrics = divide_metrics(step_metrics, self.args.accumulation_steps)
                step_loss = 0
                step_losses = []
                step_metrics = None
                # Wandb log
                if self.run is not None:
                    log_single_losses = {}
                    for i, single_loss in enumerate(print_losses):
                        log_single_losses[f'train/Task {i}/loss_batch'] = single_loss
                    """
                    self.run.log({"train/batch": epoch * len(tbar) + step, "train/loss_batch": print_loss})
                    self.run.log(flatten_metrics(print_metrics, prefix="train/", separator="/", suffix="_batch"))
                    self.run.log(log_single_losses)
                    """
                    self.run.log(
                        {"train/batch": epoch * len(tbar) + step, "train/loss_batch": print_loss} |
                        flatten_metrics(print_metrics, prefix="train/", separator="/", suffix="_batch") |
                        log_single_losses
                    )
            else:
                step_loss += cur_loss
                step_losses.append(cur_losses)
                if step_metrics is None:
                    step_metrics = cur_metrics
                else:
                    step_metrics = sum_metrics_dicts(step_metrics, cur_metrics)

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if (step + 1) % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Metrics: {print_metrics}')

                    for task, metrics in print_metrics.items():
                        for metric, value in metrics.items():
                            # print(f'task: {task}, metric: {metric}, value: {value}')
                            if value > self.step_best_metrics[task][metric]:
                                self.step_best_metrics[task][metric] = value
                                """
                                if self.step_best_metrics[task][metric] > 0.9:
                                    self.save_checkpoint(
                                        epoch,
                                        sam_model.state_dict(),
                                        describe=f'{epoch}_step_{task.replace(" ", "_")}_{metric}:{value}_best'
                                    )
                                """
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
            
        epoch_loss /= step + 1
        epoch_losses = torch.tensor(epoch_losses).detach().cpu().numpy()
        epoch_losses = np.mean(epoch_losses, axis=0)

        epoch_metrics = divide_metrics(epoch_metrics, step + 1)

        return epoch_loss, epoch_losses, epoch_metrics

    def eval_epoch(self, epoch, num_clicks):
        with torch.no_grad():
            epoch_loss = 0
            epoch_losses = []
            epoch_metrics = None
            self.model.eval()
            if self.args.multi_gpu:
                sam_model = self.model.module
            else:
                sam_model = self.model
                self.args.rank = -1

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                tbar = tqdm(self.dataloaders[1])
            else:
                tbar = self.dataloaders[1]

            for step, (image3D, mask3D, gt, path) in enumerate(tbar):

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                mask3D = mask3D.to(device).type(torch.long)
                gt = gt.to(device).type(torch.float)

                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    predictions, losses = self.batch_forward(
                        sam_model,
                        image_embedding,
                        mask3D,
                        gt,
                    )
                    loss = sum([weight * loss for weight, loss in zip(self.loss_weights, losses)])

                metrics = self.get_metrics(predictions, [gt[:, i:i + 1] for i in range(gt.size(1))])
                cur_metrics = metrics
                if epoch_metrics is None:
                    epoch_metrics = cur_metrics
                else:
                    epoch_metrics = sum_metrics_dicts(epoch_metrics, cur_metrics)

                epoch_loss += loss.item()
                epoch_losses.append(losses)

            epoch_loss /= step + 1
            epoch_losses = torch.tensor(epoch_losses).detach().cpu().numpy()
            epoch_losses = np.mean(epoch_losses, axis=0)

            epoch_metrics = divide_metrics(epoch_metrics, step + 1)

            return epoch_loss, epoch_losses, epoch_metrics
    
    def plot_result(self, plot_data, description, save_name):
        for label, data in zip(['train', 'val'], plot_data):
            plt.plot(data, label=label)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.legend()
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def train(self):
        # Freeze encoder
        if self.args.freeze_encoder:
            print("Freezing encoder")
            if not self.args.multi_gpu:
                for param in self.model.image_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.module.image_encoder.parameters():
                    param.requires_grad = False

        # Initialize counters
        trainable_params = 0
        frozen_params = 0

        # Iterate through all parameters in the model
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params += param.numel()  # Count trainable parameters
            else:
                frozen_params += param.numel()  # Count frozen parameters
        print("Trainable parameters:", trainable_params)
        print("Frozen parameters:", frozen_params)

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            self.run.watch(self.model)

        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders[0].sampler.set_epoch(epoch)
                self.dataloaders[1].sampler.set_epoch(epoch)

            num_clicks = np.random.randint(1, 21)
            epoch_loss, single_losses, epoch_metrics = self.train_epoch(epoch, num_clicks)

            # If "Detected call of `lr_scheduler.step()` before `optimizer.step()`.
            # See https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814/5
            # and https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            val_epoch_loss, val_single_losses, val_epoch_metrics = self.eval_epoch(epoch, num_clicks)

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.val_losses.append(val_epoch_loss)
                self.single_losses.append(single_losses)
                self.val_single_losses.append(val_single_losses)
                self.append_metrics(epoch_metrics)
                self.append_val_metrics(val_epoch_metrics)

                print(f'EPOCH: {epoch}, Train Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Val Loss: {val_epoch_loss}')
                logger.info(f'Epoch\t {epoch}\t : train loss: {epoch_loss}')
                logger.info(f'Epoch\t {epoch}\t : val loss: {val_epoch_loss}')

                for i, single_loss in enumerate(single_losses):
                    print(f'EPOCH: {epoch}, Train Loss Task {i}: {single_loss}')
                    logger.info(f'Epoch\t {epoch}\t : train loss task {i}: {single_loss}')
                for i, single_loss in enumerate(val_single_losses):
                    print(f'EPOCH: {epoch}, Val Loss Task {i}: {single_loss}')
                    logger.info(f'Epoch\t {epoch}\t : val loss task {i}: {single_loss}')

                for task, metrics in epoch_metrics.items():
                    for metric, value in metrics.items():
                        print(f'EPOCH: {epoch}, Train, task: {task}, metric: {metric}, value: {value}')
                        logger.info(f'EPOCH\t {epoch}\t, train\t, task: {task}, metric: {metric}, value: {value}')
                for task, metrics in val_epoch_metrics.items():
                    for metric, value in metrics.items():
                        print(f'EPOCH: {epoch}, Val, task: {task}, metric: {metric}, value: {value}')
                        logger.info(f'EPOCH\t {epoch}\t, val\t, task: {task}, metric: {metric}, value: {value}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    state_dict, 
                    describe='latest'
                )

                # save val loss best checkpoint
                if val_epoch_loss < self.best_loss:
                    self.best_loss = val_epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )

                for task, metrics in val_epoch_metrics.items():
                    for metric, value in metrics.items():
                        if value > self.best_metrics[task][metric]:
                            self.best_metrics[task][metric] = value
                            self.save_checkpoint(
                                epoch,
                                state_dict,
                                describe=f'{task.replace(" ", "_")}_{metric}_best'
                            )

                # Save plots locally
                self.plot_result([self.losses, self.val_losses], 'Weighted sum of losses', 'Loss')
                for task, (single_loss, val_single_loss) in enumerate(zip(np.stack(self.single_losses).T, np.stack(self.val_single_losses).T)):
                    self.plot_result(
                        [single_loss, val_single_loss],
                        f'Loss_Task_{task}',
                        f'Loss_Task_{task}'
                    )
                for task, metrics in self.metrics.items():
                    for metric, value in metrics.items():
                        self.plot_result(
                            [value, self.val_metrics[task][metric]],
                            f'{task}_{metric}',
                            f'{task.replace(" ", "_")}_{metric}')

            # Wandb log
            if self.run is not None:
                log_single_losses = {}
                for i, single_loss in enumerate(single_losses):
                    log_single_losses[f'train/Task {i}/loss'] = single_loss

                val_log_single_losses = {}
                for i, single_loss in enumerate(val_single_losses):
                    val_log_single_losses[f'val/Task {i}/loss'] = single_loss

                self.run.log(
                    {"train/epoch": epoch, "train/loss": epoch_loss} |
                    flatten_metrics(epoch_metrics, prefix="train/", separator="/") |
                    log_single_losses |
                    {"val/epoch": epoch, "val/loss": val_epoch_loss} |
                    flatten_metrics(val_epoch_metrics, prefix="val/", separator="/") |
                    val_log_single_losses
                )

        logger.info('=====================================================================')
        logger.info(f'Best Val loss: {self.best_loss}')
        logger.info(f'Best Val metrics: {self.best_metrics}')
        logger.info(f'Total Train loss: {self.losses}')
        logger.info(f'Total Train metric: {self.metrics}')
        logger.info(f'Total Val loss: {self.val_losses}')
        logger.info(f'Total Val metric: {self.val_metrics}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                if args.device == 'cuda':
                    args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        main_worker(idr_torch.rank, args)
        """
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
        """
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # wandb
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            mode=args.wandb_mode,
            config=args
        )
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args, run=run)
        # Train
        trainer.train()
        wandb.finish()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'),
        force=True
    )
    run = setup_run(args)

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args, run=run)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        # init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )


def setup_run(args):
    if args.log_all:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            mode=args.wandb_mode,
            group="DDP",
            config=args
        )
    else:
        if args.rank == 0:
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                mode=args.wandb_mode,
                config=args
            )
        else:
            run = None

    return run


def cleanup():
    wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
