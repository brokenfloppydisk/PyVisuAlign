from typing import TypedDict, List, Dict, Any, cast, Optional
from jsonschema import validate
import json
import os
from datetime import datetime

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


class MeasurementReferenceLine(TypedDict):
    start: List[float]
    end: List[float]
    size: float
    unit: str
    pixel_distance: float


class MeasurementLine(TypedDict):
    start: List[float]
    end: List[float]
    length: float


class SliceMeasurements(TypedDict):
    reference_line: Optional[MeasurementReferenceLine]
    measurement_lines: List[MeasurementLine]


class PyVisuAlignData(TypedDict):
    source_file: str
    measurements: Dict[str, SliceMeasurements]
    created: str
    last_modified: str


def get_sidecar_path(json_file_path: str) -> str:
    """Generate the sidecar file path for a given JSON file."""
    base = os.path.splitext(json_file_path)[0]
    return f"{base}_pyvisualign_data.json"


def load_measurement_data(json_file_path: str) -> Optional[PyVisuAlignData]:
    """
    Load measurement data from the sidecar file.
    Returns None if the sidecar file doesn't exist.
    """
    sidecar_path = get_sidecar_path(json_file_path)
    
    if not os.path.exists(sidecar_path):
        return None
    
    try:
        with open(sidecar_path, 'r') as f:
            data = json.load(f)
        return cast(PyVisuAlignData, data)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading measurement data from {sidecar_path}: {e}")
        return None


def save_measurement_data(json_file_path: str, measurements: Dict[str, SliceMeasurements]) -> bool:
    """
    Save measurement data to the sidecar file.
    Returns True if successful, False otherwise.
    """
    sidecar_path = get_sidecar_path(json_file_path)
    
    existing_data = load_measurement_data(json_file_path)
    current_time = datetime.now().isoformat() + 'Z'
    
    data: PyVisuAlignData = {
        'source_file': os.path.basename(json_file_path),
        'measurements': measurements,
        'created': existing_data['created'] if existing_data else current_time,
        'last_modified': current_time
    }
    
    try:
        with open(sidecar_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"Error saving measurement data to {sidecar_path}: {e}")
        return False


def create_slice_measurements(
    reference_line: Optional[MeasurementReferenceLine] = None,
    measurement_lines: Optional[List[MeasurementLine]] = None
) -> SliceMeasurements:
    """Create a SliceMeasurements object with optional data."""
    return {
        'reference_line': reference_line,
        'measurement_lines': measurement_lines if measurement_lines is not None else []
    }
