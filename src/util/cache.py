import os

def setup_cache(cache_dir):
    print(f'Setting up local cache at {cache_dir}')
    os.environ['HF_HOME'] = f'{cache_dir}'
    os.environ['TRANSFORMERS_CACHE'] = f'{cache_dir}/transformers'
    os.environ['HF_DATASETS_CACHE'] = f'{cache_dir}/datasets'
    os.environ['TMPDIR'] = f'{cache_dir}'