# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib

def _discover_model_files():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

names = {}

def get_all_models():
    return list(names.keys()) or _discover_model_files()

for model in _discover_model_files():
    try:
        mod = importlib.import_module('models.' + model)
    except ModuleNotFoundError as exc:
        print('Skipping model {}: {}'.format(model, exc))
        continue
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args, backbone, loss, transform):
    return names[args.model](backbone, loss, args, transform)
