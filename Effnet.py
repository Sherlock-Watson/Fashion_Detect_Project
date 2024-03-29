from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModifiedEffNet(nn.Module):
  def __init__(self, num_classes, batch_size):
    # num_classes = [7, 3, 3, 4, 6, 3]
    super().__init__()
    self.batch_size = batch_size
    self.num_classes = num_classes

    self.base_model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1) # 1280 out_features
    
    self.base_output_size = self._get_conv_output_size()
    print(f'base_output_size: {self.base_output_size}')

    self.attribute_outputs = nn.ModuleList([
            nn.Linear(self.base_output_size, i) for i in num_classes
        ])


  def forward(self, x):
        x = self.base_model(x)
        outputs = [attribute_output(x) for attribute_output in self.attribute_outputs]
        return outputs
  
# function to calculate the output size of the base CNN model
  def _get_conv_output_size(self):
      batch_size = self.batch_size
      input_tensor = torch.autograd.Variable(torch.rand(batch_size, 3, 320, 320))
      output_feat = self.base_model(input_tensor)
      n_size = output_feat.data.view(batch_size, -1).size(1)
      return n_size