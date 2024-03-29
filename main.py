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


if __name__ == '__main__':
    root = 'dataset'
    train_path = 'dataset/split/train.txt'
    val_path = 'dataset/split/val.txt'
    test_path = 'dataset/split/test.txt'

    size = 320
    crop_size = 320
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop((crop_size, crop_size)),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = FashionNet_Dataset(root, train_path, train_transform)
    val_set = FashionNet_Dataset(root, val_path, val_transform)

    batch_size = 8
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    learning_rate = 0.05
    # Parameters
    LR = 0.1
    NUM_EPOCH = 10
    WEIGHT_DECAY = 0.00001
    MOMENTUM = 0.9
    num_classes = [7, 3, 3, 4, 6, 3]

    # define Criterion
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Define model
    model = ModifiedEffNet(num_classes=num_classes, batch_size=batch_size)
    model = model.to(device)

    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters \n')

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCH)

    # Define statistic variables
    stat_train_loss = []
    stat_val_loss = []
    stat_train_acc = []
    stat_val_acc = []
    best_val_loss = 5.0

    # Start model training
    print(f'Training Model:')

    for epoch in range(NUM_EPOCH):
        start_time = time.time()
        model.train()
        training_loss = 0.0
        total_correct_train = [0] * len(num_classes)
        total_samples_train = [0] * len(num_classes)

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs_train = model(images)

            loss = sum(criterion(outputs_train[i], labels[:, i]) for i in range(len(outputs_train)))

            for i in range(len(outputs_train)):
                _, predicted = torch.max(outputs_train[i], 1)
                total_correct_train[i] += (predicted == labels[:, i]).sum().item()
                total_samples_train[i] += labels[:, i].size(0)
            
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        train_accuracies = [correct / total for correct, total in zip(total_correct_train, total_samples_train)]


        # eval
        model.eval()
        total_correct_val = [0] * len(num_classes)
        total_samples_val = [0] * len(num_classes)

        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.long().to(device)

                outputs_val = model(images)

                loss = sum(criterion(outputs_val[i], labels[:, i]) for i in range(len(outputs_val)))

                for i in range(len(outputs_val)):
                    _, predicted_val = torch.max(outputs_val[i], 1)
                    total_correct_val[i] += (predicted_val == labels[:, i]).sum().item()
                    total_samples_val[i] += labels[:, i].size(0)

                val_loss += loss.item()

        val_accuracies = [correct / total for correct, total in zip(total_correct_val, total_samples_val)]

        # Save model with best (smallest) val loss
        current_loss = val_loss / len(val_loader)
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            model_name = f'model_{current_loss:.4f}.pt'
            torch.save(model.state_dict(), model_name)

        end_time = time.time()
        epoch_secs = end_time - start_time

        stat_train_loss.append(training_loss / len(train_loader))
        stat_val_loss.append(val_loss / len(val_loader))
        stat_train_acc.append(train_accuracies)
        stat_val_acc.append(val_accuracies)

        print(f'Epoch {epoch + 1}/{NUM_EPOCH}: Training duration: {epoch_secs:.2f}s')
        print(
            f'Train Loss: {training_loss / len(train_loader):.4f} | Train Acc: {train_accuracies} | Val. Loss: {val_loss / len(val_loader):.4f} |  Val. Acc: {val_accuracies}')

        scheduler.step()

    print(f'Training Completed')
    print(f'stat_train_loss: {stat_train_loss}')
    print(f'stat_val_loss: {stat_val_loss}')
    print(f'stat_train_acc: {stat_train_acc}')
    print(f'stat_val_acc: {stat_val_acc}')
