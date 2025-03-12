from pprint import pprint
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.transforms import Compose, RandomAffine, ColorJitter, ToTensor, ToPILImage, Resize, Normalize
from dataset import VOCDataset
import torch
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def get_args():
    parser = ArgumentParser(description='FasterR-CNN for pascal VOC data')
    parser.add_argument("--root", type=str, default='./data', help='root of dataset')
    parser.add_argument("--epochs", type=int, default=20, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=4, help='number of batch size')
    parser.add_argument("--image_size", type=int, default=416, help='image size (w = h)')
    parser.add_argument("--log_path", type=str, default='./tensorboard')
    parser.add_argument("--save_path", type=str, default='./trained_models')
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help='path of trained model')
    args = parser.parse_args()

    return args


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = Compose([
        ToTensor(),
        ColorJitter(
            brightness=(0.5, 1.2),
            contrast=0.4,
            saturation=0.2,
            hue=0.2
        ),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'drop_last': True,
        'collate_fn': collate_fn
    }

    val_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'drop_last': False,
        'collate_fn': collate_fn
    }

    train_set = VOCDataset(args.root, year='2012', image_set='train', download=False, transform=train_transform)
    val_set = VOCDataset(args.root, year='2012', image_set='val', download=False, transform=val_transform)

    train_loader = DataLoader(dataset=train_set, **train_params)
    val_loader = DataLoader(dataset=val_set, **val_params)

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features, num_classes=len(train_set.categories))
    # print(model.roi_heads.box_predictor)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['best_map']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_map = -1
        if os.path.isdir(args.log_path):
            shutil.rmtree(args.log_path)
        os.mkdir(args.log_path)

    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    writer = SummaryWriter(args.log_path)
    num_iterations = len(train_loader)

    train_losses = []
    for epoch in range(start_epoch, args.epochs):
        # training
        model.train()
        train_process_bar = tqdm(train_loader, colour='cyan')
        for iteration, (images, targets) in enumerate(train_process_bar):
            # forward
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
            loss_components = model(images, targets)
            losses = sum(loss_components.values())
            train_losses.append(losses.item())
            avg_loss = np.mean(train_losses)
            train_process_bar.set_description(f"Epochs: {epoch+1}/{args.epochs}. Iteration: {iteration+1}/{num_iterations}. Loss: {avg_loss:.4f}")
            writer.add_scalar(tag='Train/Loss', scalar_value=avg_loss, global_step=epoch*num_iterations+iteration+1)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # evaluation
        model.eval()
        metric = MeanAveragePrecision(box_format='xyxy')
        val_process_bar = tqdm(val_loader, colour='blue')
        for iteration, (images, targets) in enumerate(val_process_bar):
            # forward
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]

            with torch.no_grad():
                predictions = model(images)
            metric.update(predictions, targets)

        map = metric.compute()
        writer.add_scalar('Eval/mAP', scalar_value=map['map'], global_step=epoch+1)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'best_map': best_map
        }

        # save last checkpoint
        torch.save(checkpoint, f'{args.save_path}/last.pt')

        # save best checkpoint
        if map['map'] > best_map:
            torch.save(checkpoint, f'{args.save_path}/best.pt')
            best_map = map['map']


if __name__ == '__main__':
    args = get_args()
    train(args)
