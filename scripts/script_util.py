import os
import sys
import importlib
from dataclasses import fields
import torch

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/checkpoints')

def load_checkpoint(model):
    model_path = f'{CHECKPOINT_DIR}/{model.get_name()}_checkpoint.pth'
    if os.path.exists(model_path):
        return torch.load(model_path)
    return None

def get_model_class(model_name):

  def _get_attr_case_insensitive(module, name):
    name = name.replace('_', '')
    for attr in dir(module):
      if attr.lower() == name.lower():
        return getattr(module, attr)
    return None

  model_module = importlib.import_module('src.models.' + model_name.lower())
  model_class = _get_attr_case_insensitive(model_module, model_name)
  model_config = _get_attr_case_insensitive(model_module, model_name + 'Config')

  return model_class, model_config

def get_model_from_args(args=None):
    if args is None:
        args = sys.argv[1:]
    
    model, config = get_model_class(args[0])
    
    for parameter in sys.argv[2:]:
        s = parameter.split('=')
        if len(s) != 2:
            continue
        key, value = s
        if key in [f.name for f in fields(config)]:
            if type(getattr(config, key)) == int:
                value = int(value)
            elif type(getattr(config, key)) == bool:
                value = value.lower() == 'true'
                setattr(config, key, value)
    
    return model(config)

def get_flags_from_args(args=None):
    if args is None:
        args = sys.argv[1:]
    flags = []
    for arg in sys.argv[2:]:
        if '=' in arg:
          continue
        if arg.startswith('--'):
            flags.append(arg[2:])
    return flags