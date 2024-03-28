import os

import numpy as np
import torchvision.transforms as transforms

from fashion_dataset import FashionNet_Dataset
import torch
import torchvision.models as models
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Effnet import ModifiedEffNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_properties(device))

NUM_ATTR = 6


# evaluation
def compute_avg_class_acc(gt_labels, pred_labels):
    num_attr = 6
    num_classes = [7, 3, 3, 4, 6, 3]  # number of classes in each attribute

    per_class_acc = []
    for attr_idx in range(num_attr):
        for idx in range(num_classes[attr_idx]):
            target = gt_labels[:, attr_idx]
            pred = pred_labels[:, attr_idx]
            correct = np.sum((target == pred) * (target == idx))
            total = np.sum(target == idx)
            per_class_acc.append(float(correct) / float(total))

    return sum(per_class_acc) / len(per_class_acc)


def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()
    print("Before loading data")
    i = 0
    for imgs, labels in dataloader:
        print("start data loading")
        if i == 10:
            print("i=10")
        imgs = imgs.to(device)
        labels = labels.to(device)  # [batch size, 6]

        optimizer.zero_grad()

        predictions = model(imgs)  # [batch size, 7, 6]
        loss = criterion(predictions, labels)
        loss.backward()

        optimizer.step()

        acc = accuracy(predictions, labels)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        i += 1

    train_loss = epoch_loss / len(dataloader)
    train_acc = epoch_acc / len(dataloader)

    return train_loss, train_acc


def evaluate(model, dataloader, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)  # [batch size, 6]

            predictions = model(imgs)  # [batch size, 7, 6]

            loss = criterion(predictions, labels)
            acc = accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    val_loss = epoch_loss / len(dataloader)
    val_acc = epoch_acc / len(dataloader)

    return val_loss, val_acc


def accuracy(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=1)
    predictions_wrong = torch.count_nonzero(labels - predicted_classes)
    num_predictions = torch.numel(predicted_classes)
    predictions_correct = num_predictions - predictions_wrong
    return predictions_correct.float() / num_predictions


if __name__ == '__main__':
    root = 'dataset'
    train_path = 'dataset/split/train.txt'
    val_path = 'dataset/split/val.txt'
    test_path = 'dataset/split/test.txt'

    size = 220
    crop_size = 200
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = FashionNet_Dataset(root, train_path, train_transform)
    val_set = FashionNet_Dataset(root, val_path, val_transform)

    batch_size = 4
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    learning_rate = 0.05
    # Parameters
    LR = 0.1
    NUM_EPOCH = 50
    WEIGHT_DECAY = 0.00001
    MOMENTUM = 0.9
    DROPOUT = 0.2
    num_classes = torch.tensor([7, 3, 3, 4, 6, 3])

    # define Criterion
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Define model
    model = ModifiedEffNet(num_classes, DROPOUT)
    model = model.to(device)

    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters \n')

    # Define optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCH)

    # Define statistic variables
    stat_train_loss = []
    stat_val_loss = []
    stat_train_acc = []
    stat_val_acc = []
    best_val_loss = float('inf')

    # Start model training
    print(f'Training Model:')

    for epoch in range(NUM_EPOCH):
        start_time = time.time()

        # Train model with train_dataset
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)

        # Evaluate model against val_dataset
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # Save model with best (smallest) val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = 'model.pt'
            torch.save(model.state_dict(), model_name)

        end_time = time.time()
        epoch_secs = end_time - start_time

        stat_train_loss.append(train_loss)
        stat_val_loss.append(val_loss)
        stat_train_acc.append(train_acc)
        stat_val_acc.append(val_acc)

        print(f'Epoch {epoch + 1}/{NUM_EPOCH}: Training duration: {epoch_secs:.2f}s')
        print(
            f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')

        scheduler.step()

    print(f'Training Completed')
