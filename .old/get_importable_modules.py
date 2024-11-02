import pkgutil
import importlib
import importlib.metadata
import json
import sys

SKIP_MODULES = {
    'antigravity', 'this', '__hello__', '__phello__',
    'get_importable_modules', 'inspect_module'
}

def get_importable_modules():
    modules_info = []
    for module_info in pkgutil.iter_modules():
        name = module_info.name
        if not name.startswith('_') and name not in SKIP_MODULES:
            try:
                module = importlib.import_module(name)
                try:
                    version = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    version = 'N/A'
                modules_info.append({
                    'name': name,
                    'version': version,
                    'is_package': module_info.ispkg
                })
            except:
                continue

    return json.dumps(sorted(modules_info, key=lambda x: x['name']))
