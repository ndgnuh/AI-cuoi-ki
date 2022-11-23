from ptrseg.utils import SegmentDataset, DataLoader
import torch
from ptrseg.model import FPNDBNet, SegmentLearner
import pytorch_lightning as pl

train_data = SegmentDataset(root="data/PPM-100/", mask_dir="matte")
test_data = SegmentDataset(root="data/PPM-100/", mask_dir="matte")
train_loader = DataLoader(train_data, batch_size=4)
val_loader = DataLoader(test_data, batch_size=4)

model = SegmentLearner(learning_rate=5e-5)
trainer = pl.Trainer(accelerator='cuda',
                     limit_train_batches=20,
                     limit_val_batches=5,
                     max_epochs=20)
trainer.fit(model, train_loader, val_loader)
