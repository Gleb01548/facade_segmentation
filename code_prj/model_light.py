import torch
from torch.optim.lr_scheduler import CyclicLR

import matplotlib.pyplot as plt
plt.style.use('dark_background')

import torch.nn.functional as F

from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt

from code_prj.u_net import UNET

from pytorch_optimizer import AdaBelief

import lightning.pytorch as pl


class Model_Light(pl.LightningModule):
    def __init__(self,in_channels=3, second_layer=64, 
                 out_channel=1, padding=1, downhill=3, lr=3e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.model = UNET(in_channels, second_layer, out_channel, padding=1, downhill=3)
        self.example_input_array = torch.rand(1, 3, 64, 64)
        self.valid_pred = None
        self.valid_target = None

        self.test_metric = None
        
    def forward(self, x):
        return self.model(x)        
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch['img_x']
        y = batch['img_y']
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        self.log("train_loss", loss, prog_bar=False, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = AdaBelief(self.model.parameters(), lr=self.lr)
        sch = CyclicLR(optimizer, 
                     base_lr = self.lr/40,
                     max_lr = self.lr*1.2, 
                     step_size_up = 40, 
                     mode = "triangular",
                     cycle_momentum=False)
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
            }
        }

    def validation_step(self, batch, batch_idx):
        x = batch['img_x']
        y = batch['img_y']
        prediction = self(x)
        loss = F.cross_entropy(prediction, y)
        self.log("val_loss", loss, prog_bar=False, batch_size=x.shape[0])
        
        seg_prediction = prediction.permute(0, 2, 3, 1).argmax(3)
        

        self.valid_pred = seg_prediction if self.valid_pred == None else torch.cat((self.valid_pred, seg_prediction), 0)
        self.valid_target = y if self.valid_target == None else torch.cat((self.valid_target, y), 0)
        
    def on_validation_epoch_end(self):
        metric = BinaryJaccardIndex()
        self.log("hui_metric", metric(self.valid_pred.to('cpu'), 
                                      self.valid_target.to('cpu')), prog_bar=False)
        
        self.valid_pred = None
        self.valid_target = None
    
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch['img_x']
            y = batch['img_y']
            prediction = self(x)
            
            seg_prediction = prediction.permute(0, 2, 3, 1).argmax(3)

            self.valid_pred = seg_prediction if self.valid_pred == None else torch.cat((self.valid_pred, seg_prediction), 0)
            self.valid_target = y if self.valid_target == None else torch.cat((self.valid_target, y), 0)

    def on_test_end(self):
        metric = BinaryJaccardIndex()
        test_metric = metric(self.valid_pred.to('cpu'), 
                            self.valid_target.to('cpu'))
        
        self.valid_pred = None
        self.valid_target = None
        self.test_metric = test_metric


