"""This file can be used to test various strategies"""

import reax
from reax.demos import boring_classes

trainer = reax.Trainer(strategy="ddp", devices=1, accelerator="cpu")
module = boring_classes.BoringModel()
trainer.fit(module)
