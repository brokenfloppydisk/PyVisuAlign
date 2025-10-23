from typing import Optional, Tuple
import numpy.typing as npt
import numpy as np
import logging
from scipy.spatial import Delaunay
from pyvisualign.project_data import VisualignSlice
from pyvisualign.int32slices import Int32Slices
from pyvisualign.util import profile
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
            
            y_indices = (np.arange(expected_height) * src_h / expected_height).astype(np.int32)
            x_indices = (np.arange(expected_width) * src_w / expected_width).astype(np.int32)
            
            y_indices = np.clip(y_indices, 0, src_h - 1)
            x_indices = np.clip(x_indices, 0, src_w - 1)
            
            self.scaled_atlas = self.colored_transformed_slice[y_indices[:, None], x_indices[None, :]]
            
            self.scaled_transformed_slice = self.transformed_slice[y_indices[:, None], x_indices[None, :]]
            self.scaled_wireframe = self.wireframe_slice[y_indices[:, None], x_indices[None, :]]
        else:
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
        if base_image is None:
            return
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
