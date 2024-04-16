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

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    print(sam_model)
    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(paths=img_datas, transform=tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.img_size,args.img_size,args.img_size)), # crop only object region
        tio.RandomFlip(axes=(0, 1, 2)),
    ]),
    threshold=1000)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader

class BaseTrainer:
    def __init__(self, model, dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_metric = 0.0
        self.step_best_loss = np.inf
        self.step_best_metric = 0.0
        self.losses = []
        self.metrics = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if(args.resume):
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        
    def set_loss_fn(self):
        self.loss_fn = MultipleLoss(
            loss_fns=[
                BCEWithLogitsLoss(),
                MSELoss(),
                MSELoss(),
                BCEWithLogitsLoss(),
                BCEWithLogitsLoss()
            ],
            reduction='weighted',
            weights=[1, 1, 1, 1,1]
        )
    
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
                self.best_metric = last_ckpt['best_metric']
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
            "best_metric": self.best_metric,
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
            size=(args.img_size // 4, args.img_size // 4, args.img_size // 4),
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
        loss = self.loss_fn(predictions, [gt[:, i:i+1] for i in range(gt.size(1))])

        return predictions, loss

    def get_metrics(self, predictions, gt):
        # TODO: here it is only metric for TASK 1: Classification. Complete this function for all the tasks
        total_samples = len(predictions[0])
        probabilities = torch.sigmoid(predictions[0])
        binary_predictions = torch.round(probabilities)
        correct_predictions = torch.sum(binary_predictions == gt[0])
        accuracy = correct_predictions / total_samples
        return accuracy

    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_metric = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        step_metric = 0
        # for step, (image3D, mask3D, gt) in enumerate(tbar):
        for step, (image3D, gt3D) in enumerate(tbar):
            mask3D = gt3D
            # For a sample GT is
            # [CLS, Bone Marrow (BM DS), Focal (FL), Paramedullary (PM), Extramedullary (EM)]
            gt = torch.tensor(
                [[1., 3., 2., 0., 0.]], device=self.args.device).repeat(image3D.shape[0], 1)  # GT is (N, 5)
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
                    # TODO: pred_list is not filled
                    pred_list = []

                    predictions, loss = self.batch_forward(
                        sam_model,
                        image_embedding,
                        mask3D,
                        gt,
                    )

                metric = self.get_metrics(predictions, [gt[:, i:i+1] for i in range(gt.size(1))])
                epoch_loss += loss.item()
                epoch_metric += metric.item()
                cur_loss = loss.item()
                cur_metric = metric.item()

                loss /= self.args.accumulation_steps
                
                self.scaler.scale(loss).backward()    

            if (step + 1) % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                step_loss += cur_loss  # Added this here otherwise the step_loss is not counting for the last cur_loss
                step_metric += cur_metric
                print_loss = step_loss / self.args.accumulation_steps
                print_metrics = step_metric / self.args.accumulation_steps  # This metric considers all the batches accumulated
                step_loss = 0
                step_metric = 0
            else:
                step_loss += cur_loss
                step_metric += cur_metric

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if (step + 1) % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Metrics: {print_metrics}')
                    if print_metrics > self.step_best_metric:
                        self.step_best_metric = print_metrics
                        if print_metrics > 0.9:
                            self.save_checkpoint(
                                epoch,
                                sam_model.state_dict(),
                                describe=f'{epoch}_step_metric:{print_metrics}_best'
                            )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
            
        epoch_loss /= step + 1
        epoch_metric /= step + 1

        return epoch_loss, epoch_metric, pred_list

    def eval_epoch(self, epoch, num_clicks):
        return 0
    
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()

    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_metric, pred_list = self.train_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()
        
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.metrics.append(epoch_metric)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Metric: {epoch_metric}')
                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, metric: {epoch_metric}')

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

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )
                
                # save train metric best checkpoint
                if epoch_metric > self.best_metric:
                    self.best_metric = epoch_metric
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='metric_best'
                    )

                self.plot_result(self.losses, 'Weighted sum of losses', 'Losses')
                self.plot_result(self.metrics, 'Metric', 'Metric')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best metric: {self.best_metric}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total metric: {self.metrics}')
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
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        dataloaders = get_dataloaders(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()

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
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
