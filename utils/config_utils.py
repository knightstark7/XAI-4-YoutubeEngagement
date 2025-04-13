import os
import sys
import argparse
import importlib.util
import json
from pathlib import Path

class Config:
    """A simple configuration class to replace mmcv.Config"""
    
    @staticmethod
    def fromfile(filename):
        """Load configuration from Python file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Config file {filename} does not exist")
        
        # Create empty config object
        cfg = Config()
        
        # Load Python file as module
        spec = importlib.util.spec_from_file_location("config_module", filename)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Add all attributes from module to config object
        for key in dir(config_module):
            if not key.startswith("__"):
                setattr(cfg, key, getattr(config_module, key))
                
        return cfg
    
    def __init__(self):
        """Initialize empty config"""
        pass
    
    def merge_from_dict(self, options):
        """Merge options dictionary into config"""
        for key, val in options.items():
            setattr(self, key, val)
    
    def dump(self, filename):
        """Save config to file"""
        # Get all attributes that don't start with underscore
        config_dict = {k: v for k, v in self.__dict__.items() 
                     if not k.startswith('_') and not callable(v)}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Write as JSON file
        with open(filename + '.json', 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        
        # Also save as Python file for compatibility
        with open(filename, 'w') as f:
            for key, val in config_dict.items():
                # Convert val to string representation
                if isinstance(val, str):
                    val_str = f"'{val}'"
                else:
                    val_str = str(val)
                f.write(f"{key} = {val_str}\n")

class DictAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    and append to dictionary.
    """
    @staticmethod
    def _parse_key_value(key_value):
        items = key_value.split('=', 1)
        if len(items) != 2:
            raise argparse.ArgumentError(
                None, f'Invalid format for key-value pair: {key_value}')
        key, value_str = items[0], items[1]
        
        # Try to evaluate value_str as Python code
        try:
            # If value is already quoted, keep it as string
            if (value_str.startswith('"') and value_str.endswith('"')) or \
               (value_str.startswith("'") and value_str.endswith("'")):
                value = eval(value_str)
            # Try to convert to numeric types or boolean
            elif value_str.lower() == 'true':
                value = True
            elif value_str.lower() == 'false':
                value = False
            elif value_str.isdigit():
                value = int(value_str)
            elif value_str.replace('.', '', 1).isdigit() and value_str.count('.') <= 1:
                value = float(value_str)
            else:
                # Treat as a string if it contains path characters or can't be evaluated
                value = value_str
        except:
            # If evaluation fails, use the raw string
            value = value_str
            
        return key, value

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for key_value in values:
            key, value = self._parse_key_value(key_value)
            options[key] = value
        setattr(namespace, self.dest, options) 