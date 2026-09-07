"""Automated TypeScript type and JSON Schema generator from Python Pydantic V2 contracts."""

import inspect
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin
from uuid import UUID
from datetime import datetime

# Ensure single_pass_schemas is importable
repo_root = Path(__file__).resolve().parent.parent
schemas_python_dir = repo_root / "packages" / "schemas" / "python"
if str(schemas_python_dir) not in sys.path:
    sys.path.insert(0, str(schemas_python_dir))

from pydantic import BaseModel
import single_pass_schemas as sp_schemas


def python_type_to_ts(py_type: Any) -> str:
    """Convert a Python type annotation to its TypeScript equivalent string."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        # Check for Optional[T] which is Union[T, NoneType]
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return f"{python_type_to_ts(non_none_args[0])} | null"
        return " | ".join(python_type_to_ts(a) for a in args)

    if origin in (list, List):
        item_type = python_type_to_ts(args[0]) if args else "any"
        # If item_type contains spaces or unions, wrap or format as Array<T>
        if "|" in item_type:
            return f"({item_type})[]"
        return f"{item_type}[]"

    if origin in (dict, Dict):
        key_type = python_type_to_ts(args[0]) if args else "string"
        val_type = python_type_to_ts(args[1]) if len(args) > 1 else "any"
        return f"Record<{key_type}, {val_type}>"

    if inspect.isclass(py_type):
        if issubclass(py_type, Enum):
            return py_type.__name__
        if issubclass(py_type, BaseModel):
            return py_type.__name__
        if issubclass(py_type, (int, float)):
            return "number"
        if issubclass(py_type, str):
            return "string"
        if issubclass(py_type, bool):
            return "boolean"
        if issubclass(py_type, (datetime, UUID)):
            return "string"

    if py_type is Any:
        return "any"

    return "any"


def generate_enum_ts(enum_cls: type) -> str:
    """Generate TypeScript enum declaration."""
    lines = [f"export enum {enum_cls.__name__} {{"]
    for item in enum_cls:
        val = item.value
        if isinstance(val, str):
            lines.append(f"  {item.name} = '{val}',")
        else:
            lines.append(f"  {item.name} = {val},")
    lines.append("}\n")
    return "\n".join(lines)


def generate_interface_ts(model_cls: type) -> str:
    """Generate TypeScript interface declaration from a Pydantic model."""
    lines = []
    doc = inspect.getdoc(model_cls)
    if doc:
        lines.append("/**")
        for doc_line in doc.splitlines():
            lines.append(f" * {doc_line}")
        lines.append(" */")
    lines.append(f"export interface {model_cls.__name__} {{")

    for field_name, field_info in model_cls.model_fields.items():
        ts_type = python_type_to_ts(field_info.annotation)
        is_optional = not field_info.is_required()

        # Add JSDoc for field description if present
        if field_info.description:
            lines.append(f"  /** {field_info.description} */")

        opt_marker = "?" if is_optional else ""
        lines.append(f"  {field_name}{opt_marker}: {ts_type};")

    lines.append("}\n")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Exporting Pydantic V2 Schemas to TypeScript & JSON Schema")
    print("=" * 60)

    ts_output_dir = repo_root / "packages" / "schemas" / "ts" / "src"
    json_output_dir = repo_root / "packages" / "schemas" / "json"

    ts_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect exported classes in deterministic order
    all_exports = sp_schemas.__all__
    enum_classes = []
    model_classes = []

    for name in all_exports:
        obj = getattr(sp_schemas, name)
        if inspect.isclass(obj):
            if issubclass(obj, Enum):
                enum_classes.append(obj)
            elif issubclass(obj, BaseModel):
                model_classes.append(obj)

    # 1. Generate TypeScript definitions
    ts_code = [
        "/**",
        " * Auto-generated TypeScript definitions for Single-Pass 3D Reconstruction Platform.",
        " * Generated directly from authoritative Python Pydantic V2 schemas.",
        " * DO NOT EDIT DIRECTLY. Run 'pnpm build:types' to regenerate.",
        " */\n",
    ]

    ts_code.append("// ============================================================================")
    ts_code.append("// ENUMS")
    ts_code.append("// ============================================================================\n")
    for enum_cls in enum_classes:
        ts_code.append(generate_enum_ts(enum_cls))

    ts_code.append("// ============================================================================")
    ts_code.append("// INTERFACES")
    ts_code.append("// ============================================================================\n")
    for model_cls in model_classes:
        ts_code.append(generate_interface_ts(model_cls))

    ts_index_file = ts_output_dir / "index.ts"
    ts_index_file.write_text("\n".join(ts_code), encoding="utf-8")
    print(f"Generated TypeScript declarations: {ts_index_file.relative_to(repo_root)}")

    # 2. Generate JSON Schema definitions
    for model_cls in model_classes:
        schema_dict = model_cls.model_json_schema()
        json_file = json_output_dir / f"{model_cls.__name__}.json"
        json_file.write_text(json.dumps(schema_dict, indent=2), encoding="utf-8")

    print(f"Generated {len(model_classes)} JSON Schema files in: {json_output_dir.relative_to(repo_root)}")
    print("TypeScript and JSON Schema export completed successfully.")


if __name__ == "__main__":
    main()
