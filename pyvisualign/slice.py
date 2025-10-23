from typing import Optional, Tuple
import numpy.typing as npt
import numpy as np
import logging
from scipy.spatial import Delaunay
from pyvisualign.project_data import VisualignSlice
from pyvisualign.int32slices import Int32Slices
from PIL import Image, ImageOps
import os.path

class Slice:
    def __init__(self, slice_data: VisualignSlice, atlas: Int32Slices, region_labels: dict, json_file: str) -> None:
        self.slice_index: int = 0
        self.slice_data: VisualignSlice = slice_data
        self.markers = np.array(self.slice_data["markers"])
        self.anchoring = np.array(self.slice_data["anchoring"])
        self.triangulation = self.perform_triangulation(self.markers)

        expected_width = self.slice_data["width"]
        expected_height = self.slice_data["height"]
        logging.debug(
            f"Expected dimensions: width={expected_width}, height={expected_height}"
        )

        image_path = os.path.dirname(json_file) + "/" + self.slice_data["filename"]
        logging.info(f"Loading image file: {image_path}")

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)

        logging.debug(
            f"Image dimensions: width={image.width}, height={image.height}"
        )
        # Rotate the image if necessary (width = vertical, height = horizontal for pyqt)
        if image.width == expected_width and image.height == expected_height:
            pass
            image = image.rotate(180, expand=True)
        elif image.width == expected_height and image.height == expected_width:
            pass
        else:
            logging.warning(
                "Unexpected image dimensions. Please verify the input files."
            )

        self.color_image = np.fliplr(np.array(image.convert("RGB")))
        self.gray_image = np.fliplr(np.array(image.convert("L")))

        self.atlas = atlas
        self.region_labels = region_labels  # Maps region ID to region name

        self.raw_atlas_slice: Optional[npt.NDArray[np.int32]] = None
        self.display_slice: Optional[npt.NDArray[np.int32]] = None
        self.transformed_slice: Optional[npt.NDArray[np.int32]] = None # Has region IDs
        self.normalized_transformed_slice: Optional[npt.NDArray[np.int32]] = None # For display
        self.wireframe_slice: Optional[npt.NDArray[np.int32]] = None # is also transformed
    
    def generate_slice(self) -> None:        
        if self.atlas is None or self.slice_data is None:
            logging.warning("Atlas or slice data not loaded.")
            return

        slice_origin = self.slice_data["anchoring"][0:3]
        slice_x_axis = self.slice_data["anchoring"][3:6]
        slice_y_axis = self.slice_data["anchoring"][6:9]

        logging.debug(f"Extracting atlas slice with params {(self.slice_data['anchoring'][0:9])}.")

        atlas_slice = self.atlas.get_int32_slice(
            slice_origin, slice_x_axis, slice_y_axis, grayscale=False
        )

        logging.debug(f"Atlas slice shape: {atlas_slice.shape}")
        logging.debug(f"Atlas slice data type: {atlas_slice.dtype}")
        logging.debug(f"Atlas slice min/max values: {atlas_slice.min()}/{atlas_slice.max()}")
        logging.debug(f"Atlas slice unique values count: {len(np.unique(atlas_slice))}")
        logging.debug(f"Non-zero values count: {np.count_nonzero(atlas_slice)}")

        atlas_slice = np.flipud(atlas_slice)
        self.raw_atlas_slice = atlas_slice

        self.display_slice = self.normalize_colors(self.raw_atlas_slice)
        logging.debug(f"Display slice min/max: {self.display_slice.min()}/{self.display_slice.max()}")

        # Generate the transformed atlas
        self.transformed_slice = self.generate_transformed_atlas(self.raw_atlas_slice)
        self.normalized_transformed_slice = self.normalize_colors(self.transformed_slice)
        self.wireframe_slice = self.generate_wireframe(self.transformed_slice)
    
    @staticmethod
    def normalize_colors(slice: npt.NDArray) -> npt.NDArray:
        unique_labels = np.unique(slice[slice > 0])
        normalized_slice = np.zeros_like(slice, dtype=slice.dtype)

        for i, label in enumerate(unique_labels):
            display_value = (i + 1) * (255 // len(unique_labels))
            normalized_slice[slice == label] = display_value
        
        return normalized_slice

    @staticmethod
    def generate_wireframe(slice: npt.NDArray) -> npt.NDArray:
        """Create an outline-only version of a slice"""
        outline_slice = np.zeros_like(slice, dtype=slice.dtype)

        # Detect edges using numpy array operations
        # Based on: https://stackoverflow.com/a/29488679
        
        north = slice[:-2, 1:-1]
        south = slice[2:, 1:-1]
        west = slice[1:-1, :-2]
        east = slice[1:-1, 2:]
        
        center = slice[1:-1, 1:-1]
        
        is_outline = (
            (center != north) |
            (center != south) |
            (center != west) |
            (center != east)
        )
        
        is_not_background = center != 0
        outline_mask = is_outline & is_not_background
        outline_slice[1:-1, 1:-1][outline_mask] = 255.0

        return outline_slice
    
    def generate_transformed_atlas(self, atlas: npt.NDArray) -> npt.NDArray:
        """Transform the given atlas using the triangulation to align with the brain image."""
        if self.triangulation is None:
            logging.warning("Unable to generate transformed atlas!")
            return np.zeros((100, 100), dtype=atlas.dtype)

        # Support color images: take height/width from first two dims
        output_height, output_width = atlas.shape[:2]
        transformed_atlas = np.zeros((output_height, output_width), dtype=atlas.dtype)

        # Transform each pixel from atlas space to brain image space
        # TODO: optimize with vectorized operations instead of iteration
        for y in range(atlas.shape[0]):
            for x in range(atlas.shape[1]):
                # Ignore background pixels
                if atlas[y, x] == 0:
                    continue

                transformed_x, transformed_y = self.transform_point(x, y)

                tx, ty = int(round(transformed_x)), int(round(transformed_y))

                # Check bounds and assign pixel value
                if (0 <= tx < output_width and 0 <= ty < output_height):
                    transformed_atlas[ty, tx] = atlas[y, x]

        return transformed_atlas

    def load_region_labels(self, labels_file: str) -> None:
        """Load brain region labels from the labels.txt file."""
        logging.info(f"Loading region labels from: {labels_file}")
        try:
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if (not line) or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) < 8:
                        continue

                    # Parse the line: IDX -R- -G- -B- -A-- VIS MSH LABEL
                    region_id = int(parts[0])
                    label = parts[7].strip('"')
                    self.region_labels[region_id] = label

            logging.info(f"Loaded {len(self.region_labels)} region labels")

        except Exception as e:
            logging.error(f"Failed to load region labels: {e}")
    
    @staticmethod
    def perform_triangulation(markers) -> Optional[Delaunay]:
        if markers is None:
            logging.warning("Unable to triangulate!")
            return None
        # TODO: Should this use the original modified Delaunay from VisuAlign?
        points = markers[:, 2:4]
        return Delaunay(points)

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a point from atlas space to transformed atlas space"""
        if self.triangulation is None or self.markers is None:
            return x, y

        simplex = self.triangulation.find_simplex((x, y))
        if simplex == -1:
            return x, y

        vertices = self.triangulation.simplices[simplex]
        original_points = self.markers[vertices, :2]
        transformed_points = self.markers[vertices, 2:4]

        bary_coords = self.compute_barycentric_coordinates((x, y), original_points)

        transformed_x = np.dot(bary_coords, transformed_points[:, 0])
        transformed_y = np.dot(bary_coords, transformed_points[:, 1])
        return transformed_x, transformed_y

    @staticmethod
    def compute_barycentric_coordinates(
        point: Tuple[float, float], triangle: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute barycentric coordinates for a point in a triangle"""
        A = np.array(
            [
                [triangle[0][0], triangle[1][0], triangle[2][0]],
                [triangle[0][1], triangle[1][1], triangle[2][1]],
                [1, 1, 1],
            ]
        )
        b = np.array([point[0], point[1], 1])
        return np.linalg.solve(A, b)
