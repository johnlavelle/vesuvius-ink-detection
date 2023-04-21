from training1 import get_config_model
from vesuvius.dataloader import get_train_loaders
from tqdm import tqdm
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config, model = get_config_model('output/save_model/20230413_093758/config_1.json',
                                 'output/save_model/20230413_093758/model_1.pt')

model = model.to(DEVICE)
model.eval()

all_outputs = []
total = None
tqdm_kwargs = dict(total=total, disable=False, desc='Training', position=0)
for i, datapoint in tqdm(enumerate(get_train_loaders(config)), **tqdm_kwargs):
    outputs = model(datapoint.voxels.to(DEVICE))
    center_x = np.array(list(zip(datapoint.x_start, datapoint.x_stop))).mean(axis=1)
    center_y = np.array(list(zip(datapoint.y_start, datapoint.y_stop))).mean(axis=1)
    all_outputs.append(outputs)
