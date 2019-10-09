#!/usr/bin/env python

import importlib
import inspect

modules = [
    "plasma.primitives.shots",
    "plasma.models.builder",
    "plasma.models.loader",
    "plasma.models.runner",
    "plasma.models.targets",
    "plasma.models.mpi_runner",
    "plasma.preprocessor.normalize",
    "plasma.preprocessor.preprocess",
    "plasma.utils.evaluation",
    "plasma.utils.performance",
    "plasma.utils.processing"
    ]

modules = {name: importlib.import_module(name) for name in modules}

documented = []
for moduleName, module in modules.items():
    for objName in dir(module):
        obj = getattr(module, objName)
        if (not objName.startswith("_")
                and callable(obj) and obj.__module__ == moduleName):
            print(objName, obj)
            documented.append(moduleName + "." + objName)
            if inspect.isclass(obj):
                open("docs/" + moduleName + "." + objName + ".rst", "w").write(
                    ''':orphan:

{0}
{1}

.. autoclass:: {0}
    :members:
    :special-members: __init__, __add__
    :inherited-members:
    :show-inheritance:
'''.format(moduleName + "." + objName, "="*(len(moduleName)+len(objName)+1)))
            else:
                open("docs/" + moduleName + "." + objName + ".rst",
                     "w").write(''':orphan:

{0}
{1}

.. autofunction:: {0}
'''.format(moduleName + "." + objName, "="*(len(moduleName)+len(objName)+1)))
