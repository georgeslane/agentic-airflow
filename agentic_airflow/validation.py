"""Output contracts and a small, dependency-free JSON-schema validator.

The validator supports the subset of JSON Schema that matters for tool
arguments and structured outputs: ``type`` (incl. lists of types),
``properties``, ``required``, ``additionalProperties: false``, ``items``,
``enum``, ``const``, ``anyOf``/``oneOf``, ``minItems``/``maxItems``,
``minLength``/``maxLength``, ``minimum``/``maximum``, ``pattern``.
Unknown keywords are ignored, so real-world schemas pass through safely.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from .errors import OutputValidationError, SchemaValidationError
from .models import GenerationResponse, OutputSpec

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def schema_errors(value: Any, schema: dict[str, Any] | None, path: str = "$") -> list[str]:
    """Return a list of human-readable violations (empty when valid)."""
    if not schema:
        return []
    errors: list[str] = []
    typ = schema.get("type")
    if typ is not None:
        types = typ if isinstance(typ, list) else [typ]
        if not any(_TYPE_CHECKS.get(t, lambda v: True)(value) for t in types):
            errors.append(f"{path}: expected type {'/'.join(types)}, got {type(value).__name__}")
            return errors
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']!r}")
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = schema[key]
            matches = [b for b in branches if not schema_errors(value, b, path)]
            if not matches or (key == "oneOf" and len(matches) != 1):
                errors.append(f"{path}: does not satisfy {key}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        for name, sub in props.items():
            if name in value:
                errors.extend(schema_errors(value[name], sub, f"{path}.{name}"))
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    errors.append(f"{path}: unexpected property {name!r}")
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                errors.extend(schema_errors(item, items, f"{path}[{i}]"))
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: expected at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: expected at most {schema['maxItems']} items")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")
    return errors


def validate_schema(value: Any, schema: dict[str, Any] | None) -> None:
    """Raise :class:`SchemaValidationError` if ``value`` violates ``schema``."""
    errs = schema_errors(value, schema)
    if errs:
        raise SchemaValidationError(errs)


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", re.DOTALL)


def parse_json_text(text: str) -> Any:
    """Parse JSON from model text, tolerating markdown fences and surrounding prose."""
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        stripped = m.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced JSON object/array in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise OutputValidationError("Response was not valid JSON.")


def coerce_output(response: GenerationResponse, spec: OutputSpec) -> Any:
    """Validate a final :class:`GenerationResponse` against an :class:`OutputSpec`.

    Returns the validated value or raises :class:`OutputValidationError` with a
    message suitable for feeding back to the model.
    """
    value = response.output
    text = response.text or ""
    kind = spec.kind

    if kind == "string":
        if value is None:
            return text
        return value if isinstance(value, str) else json.dumps(value)

    if kind == "list":
        if value is None:
            if not text.strip():
                raise OutputValidationError("Expected a JSON array but the response was empty.")
            value = parse_json_text(text)
        if isinstance(value, dict) and len(value) == 1 and isinstance(next(iter(value.values())), list):
            value = next(iter(value.values()))  # {"items": [...]} wrappers are common
        if not isinstance(value, list):
            raise OutputValidationError(
                f"Expected a JSON array, got {type(value).__name__}. Respond with ONLY a JSON array."
            )
        if spec.json_schema:
            _check(value, spec.json_schema)
        return value

    if kind == "json":
        if value is None:
            if not text.strip():
                raise OutputValidationError("Expected a JSON object but the response was empty.")
            value = parse_json_text(text)
        if not isinstance(value, (dict, list)):
            raise OutputValidationError(
                f"Expected a JSON object, got {type(value).__name__}. Respond with ONLY JSON."
            )
        if spec.json_schema:
            _check(value, spec.json_schema)
        return value

    if kind == "enum":
        choices = list(spec.choices or [])
        if value is None:
            raw = text.strip()
            if spec.multi:
                try:
                    value = parse_json_text(raw)
                except OutputValidationError:
                    value = [p.strip() for p in re.split(r"[,\n]", raw) if p.strip()]
            else:
                value = raw.strip().strip("\"'`")
        if spec.multi:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or not value:
                raise OutputValidationError(
                    f"Expected a JSON array with one or more of: {choices!r}."
                )
            bad = [v for v in value if v not in choices]
            if bad:
                raise OutputValidationError(
                    f"Invalid selection(s) {bad!r}. Valid values are: {choices!r}."
                )
            return list(dict.fromkeys(value))
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if value not in choices:
            raise OutputValidationError(
                f"Invalid selection {value!r}. Respond with exactly one of: {choices!r}."
            )
        return value

    if kind == "image":
        return _coerce_image(value, text)

    if kind == "file":
        return _coerce_file(value, text)

    raise OutputValidationError(f"Unknown output kind {kind!r}")


def _check(value: Any, schema: dict[str, Any]) -> None:
    errs = schema_errors(value, schema)
    if errs:
        raise OutputValidationError("Output did not match the required schema: " + "; ".join(errs))


def _coerce_image(value: Any, text: str) -> dict[str, Any]:
    if isinstance(value, (bytes, bytearray)):
        return {"kind": "image", "mime": "application/octet-stream", "data_base64": base64.b64encode(bytes(value)).decode()}
    if value is None and text.strip():
        try:
            value = parse_json_text(text)
        except OutputValidationError:
            value = text.strip()
    if isinstance(value, str):
        if os.path.isfile(value):
            return {"kind": "image", "path": os.path.abspath(value)}
        raise OutputValidationError("Expected an image payload or an existing image file path.")
    if isinstance(value, dict):
        data = value.get("data_base64") or value.get("base64")
        if data:
            try:
                base64.b64decode(data, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise OutputValidationError("data_base64 is not valid base64.") from exc
            return {"kind": "image", "mime": str(value.get("mime") or "image/png"), "data_base64": data}
        if value.get("path") and os.path.isfile(str(value["path"])):
            return {"kind": "image", "path": os.path.abspath(str(value["path"]))}
        if value.get("url"):
            return {"kind": "image", "url": str(value["url"])}
    raise OutputValidationError(
        'Expected an image as {"mime": ..., "data_base64": ...}, {"path": ...} or {"url": ...}.'
    )


def _coerce_file(value: Any, text: str) -> dict[str, Any]:
    if value is None and text.strip():
        raw = text.strip()
        try:
            value = parse_json_text(raw)
        except OutputValidationError:
            value = raw
    if isinstance(value, dict):
        value = value.get("path")
    if isinstance(value, str) and os.path.isfile(value):
        return {"kind": "file", "path": os.path.abspath(value)}
    raise OutputValidationError("Expected the path of an existing file.")
