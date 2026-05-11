import flwr as fl
import pytorch_lightning as pl
from collections import OrderedDict
import torch

import argparse

from opt import get_opts
from datasets import load_dataset, satellite
from metrics import load_loss, DepthLoss, SNerfLoss
from torch.utils.data import DataLoader
from collections import defaultdict

from rendering import render_rays
from models import load_model
import train_utils
import metrics
import os
import numpy as np
import datetime
from sat_utils import compute_mae_and_save_dsm_diff

from eval_satnerf import find_best_embbeding_for_val_image, save_nerf_output_to_images, predefined_val_ts

from main import *


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, args, logger, ckpt_callback):
        self.model = model
        self.args = args
        self.logger = logger
        self.ckpt_callback = ckpt_callback

    def get_parameters(self, config):
        print("getting parameters")
        return _get_parameters(self.model)

    def set_parameters(self, parameters):
        print("setting parameters")
        _set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        print("Inside client fit before set")
        self.set_parameters(parameters)
        print("Inside client fit after set")

        print("Trainer creation...")
        trainer = pl.Trainer(max_steps=self.args.max_train_steps,
                         max_epochs=30,
                         logger=self.logger,
                         callbacks=[self.ckpt_callback],
                         resume_from_checkpoint=self.args.ckpt_path,
                         gpus=[0],  # CUDA_VISIBLE_DEVICES set in main() makes gpu_id=0 always correct
                         auto_select_gpus=False,
                         deterministic=True,
                         benchmark=True,
                         weights_summary=None,
                         num_sanity_val_steps=2,
                         check_val_every_n_epoch=1,
                         profiler="simple")

        print("Training Started...")
        trainer.fit(self.model)
        print("Training Finished.")
        print("Train dataset length:", len(self.model.train_dataset[0]))

        return self.get_parameters(config={}), len(self.model.train_dataset[0]), {}


def _get_parameters(model):
    parameters = train_utils.get_parameters(model)
    parameters_flower = [param.detach().cpu().numpy() for param in parameters]
    return parameters_flower


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    args = get_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    torch.cuda.empty_cache()
    system = NeRF_pl(args)

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir, name=args.exp_name, default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath="{}/{}".format(args.ckpts_dir, args.exp_name),
                                                 filename="{epoch:d}",
                                                 monitor="val/psnr",
                                                 mode="max",
                                                 save_top_k=-1,
                                                 every_n_val_epochs=args.save_every_n_epochs)

    # Flower client
    print("initializing FLWR client")
    client = FlowerClient(system, args, logger, ckpt_callback)
    print("calling FLWR numpy client")
    fl.client.start_numpy_client(server_address="127.0.0.1:8085", client=client)


if __name__ == "__main__":
    main()
