import os

def setup_cache():
    os.environ['HF_HOME'] = '/cwork/wac20/tmp/cache'
    os.environ['TRANSFORMERS_CACHE'] = '/cwork/wac20/tmp/cache/transformers'
    os.environ['HF_DATASETS_CACHE'] = '/cwork/wac20/tmp/cache/datasets'
    os.environ['TMPDIR'] = '/cwork/wac20/tmp/cache'