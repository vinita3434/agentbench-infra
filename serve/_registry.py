#!/usr/bin/env python3
"""Tiny shared reader for models.yaml so the bash serve/ scripts don't have to
parse YAML themselves. Usage:

    _registry.py <key>              # print every field as KEY=VALUE
    _registry.py <key> <field>      # print a single field's value
    _registry.py --list             # print all model keys, one per line

Kept intentionally free of any harness/Pi knowledge -- serve/ only serves an
OpenAI-compatible endpoint.
"""
import pathlib
import sys

REGISTRY = pathlib.Path(__file__).resolve().parent / "models.yaml"


def load():
    try:
        import yaml
    except ImportError:
        sys.exit(
            "PyYAML is required to read models.yaml. Run serve/setup.sh first, "
            "or: pip install pyyaml"
        )
    data = yaml.safe_load(REGISTRY.read_text())
    return data.get("models", {})


def main(argv):
    models = load()
    if not argv or argv[0] in ("--list", "-l"):
        for k in models:
            print(k)
        return 0
    key = argv[0]
    if key not in models:
        sys.exit(f"unknown model key: {key!r}. Known: {', '.join(models)}")
    entry = models[key]
    if len(argv) == 1:
        for k, v in entry.items():
            print(f"{k}={v}")
        return 0
    field = argv[1]
    if field not in entry:
        sys.exit(f"model {key!r} has no field {field!r}")
    # Print empty string for empty values (bash captures "").
    v = entry[field]
    print("" if v is None else v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
