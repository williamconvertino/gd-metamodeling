import os
import sys

# Allows us to import from src
def add_src_to_path():
    src_path = os.path.join(os.path.dirname(__file__), '..')
    if src_path not in sys.path:
        sys.path.append(src_path)

# Automatically installs requirements
def install_requirements():
    if os.name == 'posix':
        os.system('pip install -r ../requirements.txt > /dev/null 2>&1')
    elif os.name == 'nt':
        os.system('pip install -r ../requirements.txt > NUL')
    else:
        raise OSError('Unsupported OS')
    
add_src_to_path()
install_requirements()