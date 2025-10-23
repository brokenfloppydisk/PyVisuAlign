from typing import TypedDict, List, Dict, Any, cast
from jsonschema import validate

class VisualignSlice(TypedDict):
    filename: str
    nr: int
    width: int
    height: int
    anchoring: List[float]
    markers: List[List[float]]

class VisualignProject(TypedDict):
    target: str
    slices: List[VisualignSlice]

visualign_schema: Dict[str, Any] = {
    "type": "object",
    "required": ["target", "slices"],
    "properties": {
        "name": {"type": "string"},
        "target": {"type": "string"},
        "target-resolution": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "number"}
        },
        "slices": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["filename", "nr", "width", "height", "anchoring", "markers"],
                "properties": {
                    "filename": {"type": "string"},
                    "nr": {"type": "integer"},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1},
                    "anchoring": {
                        "type": "array",
                        "minItems": 9,
                        "items": {"type": "number"}
                    },
                    # TODO: Markers might actually be empty, make this optional and leave the slice unscaled
                    "markers": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {"type": "number"}
                        }
                    }
                },
                "additionalProperties": True
            }
        }
    },
    "additionalProperties": True
}

def load_visualign_project(data: Dict[str, Any]) -> VisualignProject:
    """
    Validate arbitrary data and return a typed VisualignProject object.
    Raises jsonschema.ValidationError on failure.
    """
    validate(instance=data, schema=visualign_schema)
    return cast(VisualignProject, data)
