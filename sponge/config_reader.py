### Imports ###
import os
import yaml

from pathlib import Path
from typing import Optional

### Class definition ###
class ConfigReader:
    _default_configfile = 'config.yaml'

    def __init__(
        self,
        config_path: Optional[Path] = None,
    ):
        
        if config_path is None:
            # Find the config file in the module directory
            file_dir = Path(__file__).parents[0]
            config_path = os.path.join(file_dir, self._default_configfile)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)