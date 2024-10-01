
import re
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA

# import datasets_ws

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num

def get_configuration(conf):
    """Get the configuration from the name of the folder
    """
    # Split conf to get the parameters
    split_conf = conf.split("_")

    new_args = {}
    new_args["backbone"] = split_conf[0]
    new_args["aggregation"] = split_conf[1]

    if new_args["backbone"] == "cct384":
        new_args["resize"] = [384,384]

    if new_args["aggregation"] == "netvlad":
        short_sim = 0.6
    else:
        short_sim = 0.92

    if split_conf[2] == "midl":
        new_args["l2"] = "after_pool"
        radenovic = True
        short_sim = 0.95
    else:
        new_args["l2"] = "before_pool"
        radenovic = False

    return new_args, short_sim, radenovic