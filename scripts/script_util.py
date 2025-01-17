import os
import sys
import importlib
import torch
from dataclasses import fields

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/checkpoints')

def load_checkpoint(model):
    model_path = f'{CHECKPOINT_DIR}/{model.config.get_name()}_checkpoint.pth'
    if os.path.exists(model_path):
        return torch.load(model_path, weights_only=False)
    return None

def get_model_from_args(args=None):
    if args is None:
        args = sys.argv[1:]
    
    def _get_attr_case_insensitive(module, name):
        name = name.replace('_', '')
        for attr in dir(module):
            if attr.lower() == name.lower():
                return getattr(module, attr)
        return None

    model_name = args[0]

    model_module = importlib.import_module('src.models')
    model_class = _get_attr_case_insensitive(model_module, model_name)
    model_config_class = _get_attr_case_insensitive(model_module, model_name + 'Config')
    
    assert model_class is not None, f"Model class {model_name} not found"
    assert model_config_class is not None, f"Model config class {model_name}Config not found"
    
    config = model_config_class()    
    
    for parameter in sys.argv[2:]:
        s = parameter.split('=')
        if len(s) != 2:
            continue
        key, value = s
        
        def get_all_fields(x):
            fields_list = []
            for cls in x.__mro__:
                fields_list += fields(cls)
            return fields_list
        
        if key in [field.name for field in get_all_fields(model_config_class)]:
            if type(getattr(config, key)) == int:
                value = int(value)
            elif type(getattr(config, key)) == bool:
                value = value.lower() == 'true'
                setattr(config, key, value)
    
    return model_class(config)

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

import os

def setup_cache(cache_dir):
    print(f'Setting up local cache at {cache_dir}')
    os.environ['HF_HOME'] = f'{cache_dir}'
    os.environ['TRANSFORMERS_CACHE'] = f'{cache_dir}/transformers'
    os.environ['HF_DATASETS_CACHE'] = f'{cache_dir}/datasets'
    os.environ['TMPDIR'] = f'{cache_dir}'