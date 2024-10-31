#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-31 22:45:00 (ywatanabe)"
# File: ./.dotfiles/.emacs.d/lisp/python-import-manager/inspect_module.py

import inspect
import sys
from typing import Optional, Set, List, Tuple
import pandas as pd

def inspect_module(
    module: object,
    prefix: str = "",
    max_depth: int = 5,
    visited: Optional[Set[str]] = None,
    current_depth: int = 0,
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

    return pd.DataFrame(content_list, columns=['Type', 'Name', 'Docstring', 'Depth'])

# EOF
