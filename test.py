## Run the next 2 lines if the model was NOT defined and trained before
# model = ModifiedEffNet(num_classes, 0.2)
# model = model.to(device)
from Effnet import ModifiedEffNet
import numpy as np
import torchvision.transforms as transforms
from fashion_dataset import FashionNet_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLabelClassifier(nn.Module):
  def __init__(self, num_classes, num_attributes):
    super().__init__()
    self.dropout = 0.2
    self.num_classes = num_classes
    self.num_attributes = num_attributes

    self.cat1_n = num_attributes[0] # 7
    self.cat2_n = num_attributes[1] # 3
    self.cat3_n = num_attributes[2] # 3
    self.cat4_n = num_attributes[3] # 4
    self.cat5_n = num_attributes[4] # 6
    self.cat6_n = num_attributes[5] # 3

    self.effnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1) # 1536 out_features

    self.model_wo_fl = nn.Sequential(*(list(self.effnet.children())[:-1]))

    self.cat1 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = 1536, out_features = self.cat1_n),
        nn.Softmax(dim=1)
    ) # [batch size, 7]
    self.cat2 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = 1536, out_features = self.cat2_n),
        nn.Softmax(dim=1)
    ) # [batch size, 3]
    self.cat3 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = 1536, out_features = self.cat3_n),
        nn.Softmax(dim=1)
    ) # [batch size, 3]
    self.cat4 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = 1536, out_features = self.cat4_n),
        nn.Softmax(dim=1)
    ) # [batch size, 4]
    self.cat5 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = 1536, out_features = self.cat5_n),
        nn.Softmax(dim=1)
    ) # [batch size, 6]
    self.cat6 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = 1536, out_features = self.cat6_n),
        nn.Softmax(dim=1)
    ) # [batch size, 3]

  def forward(self, x):
    max_num_subclass = max(self.num_attributes) # 7
    x = self.model_wo_fl(x)
    x = torch.flatten(x,1)
    cat1 = self.cat1(x).to(device)
    if cat1.shape[1] != max_num_subclass:
        filler = torch.zeros(cat1.shape[0], (max_num_subclass - cat1.shape[1])).to(device)
        cat1 = torch.cat((cat1, filler), 1) # [batch size, 7]

    cat2 = self.cat2(x).to(device)
    if cat2.shape[1] != max_num_subclass:
        filler = torch.zeros(cat2.shape[0], (max_num_subclass - cat2.shape[1])).to(device)
        cat2 = torch.cat((cat2, filler), 1) # [batch size, 7]

    cat3 = self.cat3(x).to(device)
    if cat3.shape[1] != max_num_subclass:
        filler = torch.zeros(cat3.shape[0], (max_num_subclass - cat3.shape[1])).to(device)
        cat3 = torch.cat((cat3, filler), 1) # [batch size, 7]
      
    cat4 = self.cat4(x).to(device)
    if cat4.shape[1] != max_num_subclass:
        filler = torch.zeros(cat4.shape[0], (max_num_subclass - cat4.shape[1])).to(device)
        cat4 = torch.cat((cat4, filler), 1) # [batch size, 7]
      
    cat5 = self.cat5(x).to(device)
    if cat5.shape[1] != max_num_subclass:
        filler = torch.zeros(cat5.shape[0], (max_num_subclass - cat5.shape[1])).to(device)
        cat5 = torch.cat((cat5, filler), 1) # [batch size, 7]
      
    cat6 = self.cat6(x).to(device)
    if cat6.shape[1] != max_num_subclass:
        filler = torch.zeros(cat6.shape[0], (max_num_subclass - cat6.shape[1])).to(device)
        cat6 = torch.cat((cat6, filler), 1) # [batch size, 7]
      
    return torch.stack([cat1, cat2, cat3, cat4, cat5, cat6], dim=2) # [batch size, 7, 6]

# Test Model
model_name = "model_3.7771.pt"
num_classes = [7, 3, 3, 4, 6, 3]
batch_size = 8

model = ModifiedEffNet(num_classes=num_classes, batch_size=batch_size)

# model = MultiLabelClassifier(num_classes=6, num_attributes=num_classes)
model.to(device)

size = 320
crop_size = 320
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
root = 'dataset'
test_path = 'dataset/split/test.txt'

test_set = FashionNet_Dataset(root, test_path, test_transform)

test_loader = DataLoader(test_set, batch_size, shuffle=False)

predictions = []
# predictions = torch.Tensor().to(device) # Initialize blank tensor

model.load_state_dict(torch.load(model_name))
model.eval()
with torch.no_grad():
  for imgs, _ in test_loader:
    imgs = imgs.to(device)

    outputs_test = model(imgs) # [batch size, 7, 6]

    for i in range(len(imgs)):
        pred = [torch.argmax(output[i]).item() for output in outputs_test]
        predictions.append(pred) # [1000, 6]

    # pred = model(imgs) # [batch size, 7, 6]
    # pred = torch.argmax(pred,dim=1) # [batch size, 6]

    # predictions = torch.cat((predictions, pred), 0) # [1000, 6]

# predictions = predictions.cpu().numpy()
# predictions = predictions.astype(int)
    

# Save Prediction    
file_name = 'prediction_3.77.txt'
print(predictions)
predictions = np.array(predictions)
np.savetxt(file_name, predictions, fmt='%.1d')
print('Done \n')