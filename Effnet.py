from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModifiedEffNet(nn.Module):
  def __init__(self, num_classes, dropout):
    super().__init__()
    self.dropout = dropout
    self.num_classes = num_classes

    self.cat1_n = num_classes[0] # 7
    self.cat2_n = num_classes[1] # 3
    self.cat3_n = num_classes[2] # 3
    self.cat4_n = num_classes[3] # 4
    self.cat5_n = num_classes[4] # 6
    self.cat6_n = num_classes[5] # 3

    self.effnet = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1) # 1280 out_features
    features = 1280

    self.model_wo_fl = nn.Sequential(*(list(self.effnet.children())[:-1]))

    self.cat1 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = features, out_features = self.cat1_n),
        nn.Softmax(dim=1)
    ) # [batch size, 7]
    self.cat2 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = features, out_features = self.cat2_n),
        nn.Softmax(dim=1)
    ) # [batch size, 3]
    self.cat3 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = features, out_features = self.cat3_n),
        nn.Softmax(dim=1)
    ) # [batch size, 3]
    self.cat4 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = features, out_features = self.cat4_n),
        nn.Softmax(dim=1)
    ) # [batch size, 4]
    self.cat5 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = features, out_features = self.cat5_n),
        nn.Softmax(dim=1)
    ) # [batch size, 6]
    self.cat6 = nn.Sequential(
        nn.Dropout(p=self.dropout),
        nn.Linear(in_features = features, out_features = self.cat6_n),
        nn.Softmax(dim=1)
    ) # [batch size, 3]

  def forward(self, x):
    max_num_subclass = max(self.num_classes) # 7
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