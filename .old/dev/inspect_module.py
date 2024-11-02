#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 00:49:55 (ywatanabe)"
# File: ./python-import-manager/dev/inspect_module.py
import inspect
from typing import List, Optional, Set

import pandas as pd


def inspect_module(
    module: object,
    prefix: str = "",
    max_depth: int = 5,
    visited: Optional[Set[str]] = None,
    current_depth: int = 0,
    sort_by: List[str] = ['Type', 'Name']  # Added sorting
) -> pd.DataFrame:
    """Inspect module contents recursively."""
    if visited is None:
        visited = set()

    content_list = []

    if max_depth < 0 or module.__name__ in visited:
        return pd.DataFrame(content_list, columns=['Type', 'Name', 'Docstring', 'Depth'])

    visited.add(module.__name__)

    try:
        module_version = f" (v{module.__version__})" if hasattr(module, '__version__') else ""
        content_list.append(('M', module.__name__, module_version, current_depth))
    except Exception:
        pass

    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        if inspect.ismodule(obj):
            content_list.append(('M', full_name, "", current_depth))
            if obj.__name__.startswith(module.__name__):
                sub_df = inspect_module(obj, full_name, max_depth - 1, visited, current_depth + 1)
                content_list.extend(sub_df.values.tolist())
        elif inspect.isfunction(obj):
            content_list.append(('F', full_name, "", current_depth))
        elif inspect.isclass(obj):
            content_list.append(('C', full_name, "", current_depth))

    df = pd.DataFrame(content_list, columns=['Type', 'Name', 'Docstring', 'Depth'])
    return df.sort_values(sort_by)  # Added sorting

if __name__ == '__main__':
    import mngs
    inspect_module(mngs)


# EOF
