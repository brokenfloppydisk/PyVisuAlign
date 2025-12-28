from typing import Optional, Tuple, List
import numpy.typing as npt
import numpy as np
import logging
from scipy.spatial import Delaunay
from pyvisualign.project_data import VisualignSlice
from pyvisualign.int32slices import Int32Slices
from pyvisualign.util import profile
from pyvisualign.measurement import Measurement
from PIL import Image, ImageOps
import os.path

class Slice:
    @profile
    def __init__(self, slice_data: VisualignSlice, atlas: Int32Slices, region_labels: dict, color_map: dict, json_file: str) -> None:
        self.slice_data: VisualignSlice = slice_data
        self.markers = np.array(self.slice_data["markers"])
        self.anchoring = np.array(self.slice_data["anchoring"])
        self.triangulation = self.perform_triangulation(self.markers)

        image_path = os.path.dirname(json_file) + "/" + self.slice_data["filename"]
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)

        expected_width = self.slice_data["width"]
        expected_height = self.slice_data["height"]
        
        if image.width == expected_width and image.height == expected_height:
            image = image.rotate(180, expand=True)
        elif image.width != expected_height or image.height != expected_width:
            logging.warning("Unexpected image dimensions. Please verify the input files.")

        self.color_image = np.fliplr(np.array(image.convert("RGB")))
        self.gray_image = np.fliplr(np.array(image.convert("L")))

        self.atlas = atlas
        self.region_labels = region_labels
        self.color_map = color_map

        self.transformed_slice: Optional[npt.NDArray[np.int32]] = None
        self.colored_transformed_slice: Optional[npt.NDArray[np.uint8]] = None
        self.wireframe_slice: Optional[npt.NDArray[np.int32]] = None
        self.composite_image: Optional[npt.NDArray[np.uint8]] = None
        self.scaled_atlas: Optional[npt.NDArray[np.uint8]] = None
        self.scaled_transformed_slice: Optional[npt.NDArray[np.int32]] = None
        self.scaled_wireframe: Optional[npt.NDArray] = None
        self.atlas_rgb: Optional[npt.NDArray[np.uint8]] = None
        self.wireframe_rgb: Optional[npt.NDArray[np.uint8]] = None
        self.atlas_visible: Optional[npt.NDArray] = None
        self.wireframe_visible: Optional[npt.NDArray] = None
        
        # Measurement feature fields
        self.reference_size: Optional[float] = None
        self.reference_unit: str = "μm"
        self.reference_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.units_per_pixel: float = 0.0
        self.measurement_lines: List[Measurement] = []
        
        self.region_pixel_counts: dict[int, int] = {}
        self.region_areas: dict[int, float] = {}
        
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
    
    @profile
    def generate_slice(self) -> None:
        """Generate the transformed atlas slice and wireframe overlay."""
        if self.atlas is None or self.slice_data is None:
            logging.warning("Atlas or slice data not loaded.")
            return

        slice_origin = self.slice_data["anchoring"][0:3]
        slice_x_axis = self.slice_data["anchoring"][3:6]
        slice_y_axis = self.slice_data["anchoring"][6:9]

        logging.debug(f"Extracting atlas slice with anchoring: {self.slice_data['anchoring'][0:9]}")

        atlas_slice = self.atlas.get_int32_slice(
            slice_origin, slice_x_axis, slice_y_axis, grayscale=False
        )
        atlas_slice = np.flipud(atlas_slice)

        self.transformed_slice = self.generate_transformed_atlas(atlas_slice)
        self.colored_transformed_slice = self.apply_color_map(self.transformed_slice)
        self.wireframe_slice = self.generate_wireframe(self.transformed_slice)
        self.create_composite_image(opacity=0.5)
        
        # Calculate region pixel counts once when slice is loaded
        self.region_pixel_counts = self._calculate_region_pixel_counts()
    
    @staticmethod
    @profile
    def normalize_colors(slice: npt.NDArray) -> npt.NDArray:
        """Normalize region IDs to display values for visualization."""
        unique_labels = np.unique(slice[slice > 0])
        normalized_slice = np.zeros_like(slice, dtype=slice.dtype)
        for i, label in enumerate(unique_labels):
            display_value = (i + 1) * (255 // len(unique_labels))
            normalized_slice[slice == label] = display_value
        return normalized_slice
    
    @profile
    def apply_color_map(self, slice: npt.NDArray[np.int32]) -> npt.NDArray[np.uint8]:
        """Apply the color map from labels.txt to create an RGB image of the slice.
        
        Args:
            slice: 2D array of region IDs
            
        Returns:
            3D array (height, width, 3) with RGB colors for each region
        """
        max_region_id = max(self.color_map.keys()) if self.color_map else 0
        lookup_table = np.zeros((max_region_id + 1, 3), dtype=np.uint8)
        for region_id, color in self.color_map.items():
            lookup_table[region_id] = color
        clipped_slice = np.clip(slice, 0, max_region_id)
        return lookup_table[clipped_slice]

    @staticmethod
    @profile
    def generate_wireframe(slice: npt.NDArray) -> npt.NDArray:
        """Create an outline-only version of a slice by detecting edges."""
        outline_slice = np.zeros_like(slice, dtype=slice.dtype)
        north = slice[:-2, 1:-1]
        south = slice[2:, 1:-1]
        west = slice[1:-1, :-2]
        east = slice[1:-1, 2:]
        center = slice[1:-1, 1:-1]
        is_outline = (center != north) | (center != south) | (center != west) | (center != east)
        is_not_background = center != 0
        outline_mask = is_outline & is_not_background
        outline_slice[1:-1, 1:-1][outline_mask] = 255.0
        return outline_slice
    
    @profile
    def generate_transformed_atlas(self, atlas: npt.NDArray) -> npt.NDArray:
        """Transform the atlas using triangulation to align with the brain image."""
        if self.triangulation is None:
            logging.warning("Unable to generate transformed atlas!")
            return np.zeros((100, 100), dtype=atlas.dtype)

        output_height, output_width = atlas.shape[:2]
        transformed_atlas = np.zeros((output_height, output_width), dtype=atlas.dtype)
        y_coords, x_coords = np.where(atlas != 0)
        
        if len(x_coords) == 0:
            return transformed_atlas
        
        pixel_values = atlas[y_coords, x_coords]
        points = np.column_stack([x_coords, y_coords])
        simplex_indices = self.triangulation.find_simplex(points)
        
        inside_mask = simplex_indices != -1
        outside_mask = ~inside_mask
        
        # Handle points outside triangulation (identity mapping)
        if np.any(outside_mask):
            outside_points = points[outside_mask]
            outside_values = pixel_values[outside_mask]
            tx = np.round(outside_points[:, 0]).astype(int)
            ty = np.round(outside_points[:, 1]).astype(int)
            valid = (tx >= 0) & (tx < output_width) & (ty >= 0) & (ty < output_height)
            transformed_atlas[ty[valid], tx[valid]] = outside_values[valid]
        
        if np.any(inside_mask):
            inside_points = points[inside_mask]
            inside_values = pixel_values[inside_mask]
            inside_simplices = simplex_indices[inside_mask]
            
            vertices = self.triangulation.simplices[inside_simplices]
            
            orig_pts = self.markers[vertices, :2]
            trans_pts = self.markers[vertices, 2:4]
            
            # barycentric coordinate computation
            # For each point, solve: [v0.x v1.x v2.x] [w0]   [p.x]
            #                        [v0.y v1.y v2.y] [w1] = [p.y]
            #                        [  1    1    1 ] [w2]   [ 1 ]

            A = np.zeros((len(inside_points), 3, 3))
            A[:, 0, :] = orig_pts[:, 0, :]
            A[:, 1, :] = orig_pts[:, 1, :]
            A[:, 2, :] = orig_pts[:, 2, :]
            A = A.transpose(0, 2, 1)
            A[:, 2, :] = 1.0
            
            b = np.ones((len(inside_points), 3))
            b[:, 0] = inside_points[:, 0]
            b[:, 1] = inside_points[:, 1]
            
            bary_coords = np.linalg.solve(A, b)
            
            transformed_x = np.sum(bary_coords * trans_pts[:, :, 0], axis=1)
            transformed_y = np.sum(bary_coords * trans_pts[:, :, 1], axis=1)
            
            tx = np.round(transformed_x).astype(int)
            ty = np.round(transformed_y).astype(int)
            valid = (tx >= 0) & (tx < output_width) & (ty >= 0) & (ty < output_height)
            transformed_atlas[ty[valid], tx[valid]] = inside_values[valid]

        return transformed_atlas

    @profile
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
                    region_id = int(parts[0])
                    label = parts[7].strip('"')
                    self.region_labels[region_id] = label
            logging.info(f"Loaded {len(self.region_labels)} region labels")
        except Exception as e:
            logging.error(f"Failed to load region labels: {e}")
    
    @staticmethod
    @profile
    def perform_triangulation(markers) -> Optional[Delaunay]:
        if markers is None:
            logging.warning("Unable to triangulate!")
            return None
        return Delaunay(markers[:, 2:4])

    @profile
    def prepare_overlay_data(self) -> None:
        """Prepare cached data for faster composite image creation."""
        if (self.colored_transformed_slice is None or 
            self.wireframe_slice is None or
            self.transformed_slice is None or
            self.color_image is None):
            logging.warning("Cannot prepare overlay - missing required images")
            return

        expected_height = self.slice_data["height"]
        expected_width = self.slice_data["width"]
        
        if (self.colored_transformed_slice.shape[:2] != (expected_height, expected_width)):
            src_h, src_w = self.colored_transformed_slice.shape[:2]
            self.scale_x = src_w / expected_width
            self.scale_y = src_h / expected_height
            
            y_indices = (np.arange(expected_height) * self.scale_y).astype(np.int32)
            x_indices = (np.arange(expected_width) * self.scale_x).astype(np.int32)

            y_indices = np.clip(y_indices, 0, src_h - 1)
            x_indices = np.clip(x_indices, 0, src_w - 1)
            
            self.scaled_atlas = self.colored_transformed_slice[y_indices[:, None], x_indices[None, :]]
            
            self.scaled_transformed_slice = self.transformed_slice[y_indices[:, None], x_indices[None, :]]
            self.scaled_wireframe = self.wireframe_slice[y_indices[:, None], x_indices[None, :]]
        else:
            self.scale_x, self.scale_y = 1, 1
            self.scaled_atlas = self.colored_transformed_slice
            self.scaled_transformed_slice = self.transformed_slice
            self.scaled_wireframe = self.wireframe_slice

        self.atlas_visible = np.any(self.scaled_atlas > 0, axis=-1)
        self.wireframe_visible = self.scaled_wireframe > 0
        self.atlas_rgb = self.scaled_atlas
        self.wireframe_rgb = np.where(
            self.scaled_wireframe[..., None] > 0, 
            255, 
            0
        ).astype(np.uint8)

    @profile
    def create_composite_image(self, opacity: float = 0.5, base_image: Optional[npt.NDArray[np.uint8]] = None) -> None:
        """Create a composite image by overlaying the transformed atlas on the brain image.
        
        Args:
            opacity: Float between 0 and 1 indicating atlas opacity (default: 0.5)
            base_image: Optional base image to use instead of self.color_image (for grayscale mode)
        """
        if base_image is None:
            base_image = self.color_image
        if base_image.ndim == 2:
            base_image = np.stack([base_image] * 3, axis=-1)
        
        needs_init = (not hasattr(self, '_comp_state') or 
                      not hasattr(self, '_comp_buffer') or 
                      self._comp_buffer.shape != base_image.shape)
        
        if needs_init:
            self.prepare_overlay_data()
            self._comp_buffer = np.empty_like(base_image)
            self._has_atlas = self.atlas_visible is not None and np.any(self.atlas_visible)
            self._has_wireframe = self.wireframe_visible is not None and np.any(self.wireframe_visible)
            if self._has_atlas and self.atlas_visible is not None:
                self._atlas_mask_3d = np.stack([self.atlas_visible] * 3, axis=2)
            if self._has_wireframe and self.wireframe_visible is not None:
                self._wireframe_mask_3d = np.stack([self.wireframe_visible] * 3, axis=2)
            self._comp_state = True
        
        composite = self._comp_buffer
        np.copyto(composite, base_image)
        
        if opacity > 0 and self._has_atlas and self.atlas_rgb is not None:
            base_pixels = composite[self._atlas_mask_3d].astype(np.float32)
            overlay_pixels = self.atlas_rgb[self._atlas_mask_3d].astype(np.float32)
            blended = base_pixels * (1.0 - opacity) + overlay_pixels * opacity
            composite[self._atlas_mask_3d] = blended.astype(np.uint8)
        
        if self._has_wireframe and self.wireframe_rgb is not None:
            base_pixels = composite[self._wireframe_mask_3d].astype(np.float32)
            num_pixels = len(base_pixels) // 3
            light_blue_repeated = np.tile([135.0, 206.0, 250.0], num_pixels).astype(np.float32)
            blended = base_pixels * 0.4 + light_blue_repeated * 0.6
            composite[self._wireframe_mask_3d] = blended.astype(np.uint8)
        
        
        self.composite_image = composite

    def _display_to_full_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Convert display (scaled) coordinates to full resolution coordinates."""
        return (x * self.scale_x, y * self.scale_y)
    
    def _full_to_display_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Convert full resolution coordinates to display (scaled) coordinates."""
        return (x / self.scale_x, y / self.scale_y)
    
    def set_reference_measurement(self, size: float, unit: str) -> None:
        """Set the reference measurement size and unit."""
        self.reference_size = size
        self.reference_unit = unit
    
    def set_reference_line(self, start: Tuple[float, float], end: Tuple[float, float], from_saved: bool = False) -> None:
        """Set the reference line for measurement calibration.
        
        Args:
            start: Start point coordinates
            end: End point coordinates
            from_saved: If True, coordinates are in full resolution (from saved data).
                       If False, coordinates are in display space (from user clicks).
        """
        if from_saved:
            start_full = start
            end_full = end
        else:
            start_full = self._display_to_full_coords(start[0], start[1])
            end_full = self._display_to_full_coords(end[0], end[1])
        
        self.reference_line = (start_full, end_full)
        
        pixel_distance = self.calculate_pixel_distance(start_full, end_full)
        if pixel_distance > 0 and self.reference_size is not None:
            self.units_per_pixel = self.reference_size / pixel_distance
        else:
            self.units_per_pixel = 0.0
        
        self.update_region_areas()
    
    def get_reference_line_display(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get the reference line in display coordinates."""
        if self.reference_line is None:
            return None
        
        start_full, end_full = self.reference_line
        start_display = self._full_to_display_coords(start_full[0], start_full[1])
        end_display = self._full_to_display_coords(end_full[0], end_full[1])
        return (start_display, end_display)
    
    def calculate_pixel_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate the pixel distance between two points."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def add_measurement_line(self, start: Tuple[float, float], end: Tuple[float, float]) -> None:
        """Add a measurement line and calculate its length based on the reference.
        Coordinates are expected in display space and stored in full resolution space.
        """
        if self.reference_line is None or self.reference_size is None:
            logging.warning("Cannot add measurement line: reference not set")
            return
        
        start_full = self._display_to_full_coords(start[0], start[1])
        end_full = self._display_to_full_coords(end[0], end[1])
        
        pixel_distance = self.calculate_pixel_distance(start_full, end_full)
        
        if self.units_per_pixel > 0:
            length = pixel_distance * self.units_per_pixel
        else:
            length = 0.0
        
        self.measurement_lines.append(Measurement(
            start=np.array(start_full),
            end=np.array(end_full),
            length=length
        ))
    
    def get_measurement_lines_display(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """Get all measurement lines in display coordinates.
        
        Returns:
            List of tuples: (start_display, end_display, length)
        """
        result = []
        for measurement in self.measurement_lines:
            start_display = self._full_to_display_coords(
                measurement.start[0], measurement.start[1]
            )
            end_display = self._full_to_display_coords(
                measurement.end[0], measurement.end[1]
            )
            result.append((start_display, end_display, measurement.length))
        return result
    
    def remove_measurement_line(self, x: float, y: float, threshold: float = 10.0) -> bool:
        """Remove a measurement line near the given point.
        Coordinates are expected in display space.
        
        Returns True if a line was removed, False otherwise.
        """
        x_full, y_full = self._display_to_full_coords(x, y)
        
        threshold_full = threshold * self.scale_x
        
        for i, measurement in enumerate(self.measurement_lines):
            p = np.array([x_full, y_full])
            p1 = measurement.start
            p2 = measurement.end
            
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            
            if line_len_sq == 0:
                dist = np.linalg.norm(p - p1)
            else:
                # Project point onto line
                t = np.clip(np.dot(p - p1, line_vec) / line_len_sq, 0.0, 1.0)

                closest = p1 + t * line_vec
                dist = np.linalg.norm(p - closest)
            
            if dist <= threshold_full:
                del self.measurement_lines[i]
                return True
        
        return False
    
    def clear_all_measurements(self) -> None:
        """Clear all measurement lines."""
        self.measurement_lines.clear()
    
    def _calculate_region_pixel_counts(self) -> dict[int, int]:
        """
        Calculate the number of pixels for each region in the transformed slice.
        Uses the full resolution transformed slice (not the scaled display version).
        This is called once during slice generation.
        
        Returns:
            Dictionary mapping region_id -> pixel_count
        """
        if self.transformed_slice is None:
            return {}
        
        unique_regions, counts = np.unique(self.transformed_slice, return_counts=True)
        
        region_pixel_counts = {}
        for region_id, count in zip(unique_regions, counts):
            if region_id > 0:
                region_pixel_counts[int(region_id)] = int(count)
        
        return region_pixel_counts
    
    def update_region_areas(self) -> dict[int, float]:
        """
        Calculate the area of each region in the slice using the reference line scale.
        
        Returns:
            Dictionary mapping region_id -> area (in reference_unit^2)
            Returns empty dict if no reference line is set.
        """
        if (self.reference_line is None or self.reference_size is None or 
            self.transformed_slice is None or self.units_per_pixel == 0):
            self.region_areas = {}
            return {}
        
        area_per_pixel = self.units_per_pixel ** 2
        
        region_areas = {}
        for region_id, pixel_count in self.region_pixel_counts.items():
            region_areas[region_id] = pixel_count * area_per_pixel
        
        self.region_areas = region_areas
        return region_areas
    
    def get_region_area_data(self) -> List[Tuple[int, str, int, float]]:
        """
        Get comprehensive region area data for CSV export.
        
        Returns:
            List of tuples: (region_id, region_name, pixel_count, area)
            Sorted by region_id
        """
        pixel_counts = self.region_pixel_counts
        
        data = []
        for region_id in sorted(pixel_counts.keys()):
            region_name = self.region_labels.get(region_id, f"Unknown_{region_id}")
            pixel_count = pixel_counts[region_id]
            area = self.region_areas.get(region_id, 0.0)
            data.append((region_id, region_name, pixel_count, area))
        
        return data


