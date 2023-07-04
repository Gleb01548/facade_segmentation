import os
import pandas as pd
import cv2

import albumentations as A

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.tuner import Tuner

from code_prj.data import BuildDataset
from code_prj.model_light import Model_Light

device = 'cuda:0' if torch.cuda.is_available else 'cpu'

current_dir = os.getcwd().split('/')
current_dir[-1]  = 'data/CMP_facade_DB_base/base'

folder = '/'.join(current_dir)

# train = [i for i in listdir(folder) if i.endswith('.jpg')]
# train, test = train_test_split(train, test_size=0.3, random_state=4, shuffle=True)
# test, val = train_test_split(test, test_size=0.5, random_state=4, shuffle=True)
# pd.DataFrame(train, columns=['values']).to_csv('data/train.csv')
# pd.DataFrame(val, columns=['values']).to_csv('data/val.csv')
# pd.DataFrame(test, columns=['values']).to_csv('data/test.csv')

train = pd.read_csv('data/train.csv')['values'].to_list()
val = pd.read_csv('data/val.csv')['values'].to_list()
test = pd.read_csv('data/test.csv')['values'].to_list()

transform = A.Compose([
    A.PadIfNeeded(min_height=1024, min_width=1024, p=1), 
    A.RandomCrop(width=720, height=720)
])

transform_other = A.Compose([
    A.Rotate(limit=80, p=0.9, border_mode = cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2)
])

transform_val = A.Compose([
    A.PadIfNeeded(min_height=1024, min_width=1024, p=1)
])


train_data = BuildDataset(folder, train, transform, transform_other)
val = BuildDataset(folder, val, transform, tr_chance=-1)
test_data = BuildDataset(folder, test, transform, tr_chance=-1)

train_load = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=24)
valid_load = DataLoader(val, batch_size=8, num_workers=24)
test_load = DataLoader(test_data, batch_size=8, num_workers=24)


checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoint',
        filename='checkpoint'
    )
comet_logger = CometLogger(api_key="ddEXKrGdl7ryYW2YWc6A6mWNd")
lr_logger = LearningRateMonitor(logging_interval='epoch')

model = Model_Light(3, 64, 2)
trainer = pl.Trainer(callbacks=[checkpoint_callback, lr_logger],
                    accelerator='cuda', 
                    devices=[0],
                    max_epochs=300, 
                    logger=comet_logger
                    )

tuner = Tuner(trainer)
tuner.lr_find(model, train_load)

trainer.fit(model, train_load, valid_load)