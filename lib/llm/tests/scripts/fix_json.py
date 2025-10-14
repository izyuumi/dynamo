#!/usr/bin/env python3
"""
Transform a chat completion stream JSON file in-place.

Changes applied:
- Move top-level "normal_content", "reasoning_content", and "tool_calls"
  under a new top-level key "expected_output".
- Rename top-level key "data" to "input_stream".
- Serialize output so each element in "input_stream" occupies exactly one line.

Usage:
    python fix_json.py /path/to/chat_completion_stream_*.json
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple


FIELDS_TO_NEST = ["normal_content", "reasoning_content", "tool_calls"]


def build_expected_output(source: Dict[str, Any]) -> Dict[str, Any]:
    expected: Dict[str, Any] = {}
    for field in FIELDS_TO_NEST:
        if field in source:
            expected[field] = source[field]
    return expected


def transform_top_level(obj: Dict[str, Any]) -> OrderedDict:
    original_keys: List[str] = list(obj.keys())
    expected_output_map: Dict[str, Any] = build_expected_output(obj)

    transformed: "OrderedDict[str, Any]" = OrderedDict()
    expected_inserted = False

    for key in original_keys:
        if key in FIELDS_TO_NEST:
            if not expected_inserted and expected_output_map:
                transformed["expected_output"] = expected_output_map
                expected_inserted = True
            continue

        if key == "data":
            transformed["input_stream"] = obj[key]
            continue

        transformed[key] = obj[key]

    if expected_output_map and not expected_inserted:
        transformed["expected_output"] = expected_output_map

    return transformed


def dump_with_one_line_list(
    obj: Dict[str, Any], list_key: str = "input_stream", indent: int = 2
) -> str:
    lines: List[str] = ["{"]
    keys: List[str] = list(obj.keys())

    for idx, key in enumerate(keys):
        is_last_key = idx == len(keys) - 1
        key_json = json.dumps(key, ensure_ascii=False)
        value = obj[key]

        if key == list_key and isinstance(value, list):
            lines.append(" " * indent + f"{key_json}: [")
            for i, item in enumerate(value):
                item_json = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
                trailing = "," if i < len(value) - 1 else ""
                lines.append(" " * (indent * 2) + item_json + trailing)
            closing = "]" + ("" if is_last_key else ",")
            lines.append(" " * indent + closing)
            continue

        value_json = json.dumps(value, indent=indent, ensure_ascii=False)
        if "\n" in value_json:
            base = " " * indent
            split_lines = value_json.splitlines()
            first_line = split_lines[0]
            rest_lines = split_lines[1:]
            suffix = "" if is_last_key else ","
            # Write the key line with exactly one space after ':' and no extra spaces before '{'
            lines.append(base + f"{key_json}: " + first_line)
            # Indent subsequent lines by one indentation level beyond the key line
            for i, ln in enumerate(rest_lines):
                trailing = suffix if i == len(rest_lines) - 1 else ""
                lines.append(base + (" " * indent) + ln + trailing)
        else:
            suffix = "" if is_last_key else ","
            lines.append(" " * indent + f"{key_json}: {value_json}" + suffix)

    lines.append("}")
    return "\n".join(lines) + "\n"


def process_file(path: Path) -> Tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return False, f"Failed to read JSON: {exc}"

    if not isinstance(data, dict):
        return False, "Top-level JSON must be an object"

    transformed = transform_top_level(data)
    output = dump_with_one_line_list(transformed, list_key="input_stream", indent=2)

    try:
        with path.open("w", encoding="utf-8") as f:
            f.write(output)
    except Exception as exc:
        return False, f"Failed to write JSON: {exc}"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix and reformat stream JSON file in-place")
    parser.add_argument("json_file", type=str, help="Path to the JSON file to transform")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        raise SystemExit(f"Path does not exist: {json_path}")
    if not json_path.is_file():
        raise SystemExit(f"Not a file: {json_path}")

    ok, msg = process_file(json_path)
    if not ok:
        raise SystemExit(msg)


if __name__ == "__main__":
    main()


