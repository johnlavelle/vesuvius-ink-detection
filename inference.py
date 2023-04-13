from training import get_config_model, train_loader_regular_z
from tqdm import tqdm
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config, model = get_config_model('output/save_model/20230413_093758/config_1.json',
                                 'output/save_model/20230413_093758/model_1.pt')

model.eval()

all_outputs = []
total = None
tqdm_kwargs = dict(total=total, disable=False, desc='Training', position=0)
for i, datapoint in tqdm(enumerate(train_loader_regular_z), **tqdm_kwargs):
    outputs = model(datapoint.voxels.to(DEVICE))
    all_outputs.append(outputs)
