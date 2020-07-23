import json
import os
from collections import namedtuple
from datetime import datetime
from shutil import make_archive


def load_json(path):
    with open(path) as f:
        json_contents = json.load(f)
    return json_contents


def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj


def get_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config


def update_config(config, args):
    for entry in config:
        if hasattr(args, entry):
            if eval("args.{}".format(entry)) is not None:
                config[entry] = eval("args.{}".format(entry))
    return config


def copy_source(code_directory, model_dir):
  now = datetime.now().strftime('%Y-%m-%d')
  make_archive(os.path.join(model_dir, "code_%s.tar.gz" % now), 'tar', code_directory)


