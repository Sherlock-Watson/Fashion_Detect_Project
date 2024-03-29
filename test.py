## Run the next 2 lines if the model was NOT defined and trained before
# model = ModifiedEffNet(num_classes, 0.2)
# model = model.to(device)
from Effnet import ModifiedEffNet
import numpy as np

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Test Model
model_name = "model_3.8044"
num_classes = [7, 3, 3, 4, 6, 3]

model = ModifiedEffNet(num_classes=num_classes, batch_size=batch_size)
predictions = torch.Tensor().to(device)
model.load_state_dict(torch.load(model_name))
model.eval()
with torch.no_grad():
  for imgs in test_dataloader:
    imgs = imgs.to(device)

    pred = model(imgs) # [batch size, 7, 6]
    pred = torch.argmax(pred,dim=1) # [batch size, 6]

    predictions = torch.cat((predictions, pred), 0) # [1000, 6]

# Save Prediction    
predictions = predictions.cpu().numpy()
predictions = predictions.astype(int)
file_name = 'prediction.txt' 
np.savetxt(file_name, predictions, fmt='%.1d')
print('Done \n')
model_name_temp = 'model_3.8044.pt'
model.load_state_dict(torch.load(model_name_temp))
model.eval()

predictions = torch.Tensor().to(device) # Initialize blank tensor

with torch.no_grad():
  for imgs in test_dataloader:
    imgs = imgs.to(device)

    pred = model(imgs) # [batch size, 7, 6]
    pred = torch.argmax(pred,dim=1) # [batch size, 6]

    predictions = torch.cat((predictions, pred), 0) # [1000, 6]

predictions = predictions.cpu().numpy()
predictions = predictions.astype(int)

## Run next 2 lines only if want to save the predictions as a .txt file
# file_name = 'prediction.txt' 
# np.savetxt(file_name, predictions, fmt='%.1d')

print('Done \n')

# Load results
previous_saved_pred = "prediction_LR0.1_EPOCH50_B32_WD1e-5.txt"
previous_saved_pred = np.loadtxt(previous_saved_pred, dtype=int)

# Compare loaded results with predictions from saved model, output should be 0
num_different_pred = np.count_nonzero((predictions - previous_saved_pred))
print(num_different_pred)