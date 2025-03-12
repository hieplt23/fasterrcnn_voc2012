import cv2
import torch
import numpy as np
from pprint import pprint
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, RandomAffine, ColorJitter, ToTensor, ToPILImage, Resize, Normalize


class VOCDataset(VOCDetection):
    def __init__(self, root="./data", year='2012', image_set='train', download=False, transform=None):
        super().__init__(root=root, year=year, image_set=image_set, download=download, transform = transform)
        self.categories = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
        ]

    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        _, new_h, new_w = image.shape
        old_w, old_h = int(target['annotation']['size']['width']), int(target['annotation']['size']['height'])

        bboxes = []
        labels = []
        output = {}
        for obj in target['annotation']['object']:
            label = self.categories.index(obj['name'])
            labels.append(label)

            xmin = float(obj['bndbox']['xmin'])/old_w*new_w
            ymin = float(obj['bndbox']['ymin'])/old_h*new_h
            xmax = float(obj['bndbox']['xmax'])/old_w*new_w
            ymax = float(obj['bndbox']['ymax'])/old_h*new_h
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            bboxes.append(bbox)
        output['boxes'] = torch.FloatTensor(bboxes)
        output['labels'] = torch.LongTensor(labels)
        return image, output

if __name__ == '__main__':
    train_transform = Compose([
        ToTensor(),
        ColorJitter(
            brightness=(0.5, 1.2),
            contrast=0.5,
            saturation=0.2,
            hue=0.1
        ),
        Resize((416, 416)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    dataset = VOCDataset('./data', year='2012', image_set='trainval', download=False, transform=train_transform)
    print(len(dataset))
    image, target = dataset[10279]
    bboxes = target['boxes']
    labels = target['labels']
    print(image.shape)
    print(bboxes.shape)
    print(labels.shape)

    # for anno, label in zip(bboxes, labels):
    #     xmin, ymin, xmax, ymax = anno
    #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=1)
    #     cv2.putText(image, dataset.categories[label], (xmin, ymin-5 if ymin > 10 else 15), fontFace=cv2.FONT_ITALIC, fontScale=0.8, color=(0,0,244), thickness=2)
    #
    #
    # cv2.imshow('test', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)