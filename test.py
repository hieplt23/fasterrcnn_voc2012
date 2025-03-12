from argparse import ArgumentParser
import torch
import cv2
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, \
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import os
import numpy as np
from torchvision.ops import nms

def get_args():
    parser = ArgumentParser(description='FasterR-CNN for pascal VOC data')
    parser.add_argument("--image_path", type=str, default='./data/test_images/test_1.jpg', help='root of dataset')
    parser.add_argument("--image_size", type=int, default=416, help='number of epochs')
    parser.add_argument("--checkpoint", "-c", type=str, default='./trained_models/best.pt', help='path of trained model')
    args = parser.parse_args()

    return args

def deploy(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    categories = [
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
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=len(categories))
    model.to(device)

    if not os.path.isfile(args.checkpoint):
        print('Not found checkpoint!')
        exit(0)

    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f'mAP: {checkpoint['best_map']}')

    original_image = cv2.imread(args.image_path)
    old_h, old_w, _ = original_image.shape
    test_image = cv2.resize(original_image, (args.image_size, args.image_size))
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB).astype(np.float64)
    test_image /= 255.
    test_image -= np.array([0.485, 0.456, 0.406])
    test_image /= np.array([0.229, 0.224, 0.225])
    test_image = test_image.transpose(2, 0 ,1)
    test_image = [torch.FloatTensor(test_image)]

    model.eval()
    with torch.no_grad():
        results = model(test_image)

    results = results[0]
    boxes = results['boxes']
    labels = results['labels']
    scores = results['scores']

    indices = nms(boxes, scores, 0.5)
    for indice in indices:
        if scores[indice] > 0.2:
            xtl, ytl, xbr, ybr = boxes[indice]
            xtl = int(float(xtl.item())/args.image_size*old_w)
            ytl = int(float(ytl.item())/args.image_size*old_h)
            xbr = int(float(xbr.item())/args.image_size*old_w)
            ybr = int(float(ybr.item())/args.image_size*old_h)
            label_confident_text = f'{categories[labels[indice]]} {scores[indice]:.2f}'
            cv2.rectangle(original_image, (xtl, ytl), (xbr, ybr), color=(0,255,0), thickness=1)
            # cv2.rectangle(original_image, (xtl, ytl-20 if ytl > 20 else 2), (xtl+100, ytl if ytl > 20 else 20), color=(255,255,255), thickness=-1)
            cv2.putText(original_image, label_confident_text, (xtl, ytl-5 if ytl > 10 else 15), fontFace=cv2.FONT_ITALIC, fontScale=0.6, color=(0,0,200), thickness=2)
            print(scores[indice])
    cv2.imshow('test', original_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = get_args()
    deploy(args)