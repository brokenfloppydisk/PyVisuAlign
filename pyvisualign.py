import sys
import json
import numpy as np
import numpy.typing as npt
import logging
import argparse
from typing import List, Tuple, Optional, Any, Dict
from jsonschema import ValidationError
from project_data import load_visualign_project, VisualignSlice
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QPushButton,
    QWidget,
    QCheckBox,
    QLabel,
)
import os.path
from pyqtgraph import GraphicsLayoutWidget, GraphicsScene, ImageItem, PlotDataItem, ViewBox
from scipy.spatial import Delaunay
from PIL import Image, ImageOps
from typing import List, Tuple, Optional
from int32slices import Int32Slices


class VisuAlignApp(QMainWindow):
    def __init__(self, json_file: Optional[str] = None, debug: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("VisuAlign Viewer")
        self.setGeometry(100, 100, 1200, 800)

        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # Main layout
        self.central_widget: QWidget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout: QVBoxLayout = QVBoxLayout(self.central_widget)

        # Control panel
        self.control_panel: QWidget = QWidget()
        self.control_layout: QHBoxLayout = QHBoxLayout(self.control_panel)
        self.outline_checkbox: QCheckBox = QCheckBox("Show Region Outlines Only")
        self.outline_checkbox.stateChanged.connect(self.toggle_outline_mode)
        self.control_layout.addWidget(self.outline_checkbox)

        # grayscale toggle checkbox
        self.grayscale_checkbox: QCheckBox = QCheckBox("Grayscale Image")
        self.grayscale_checkbox.setChecked(False)
        self.grayscale_checkbox.stateChanged.connect(self.toggle_grayscale)
        self.control_layout.addWidget(self.grayscale_checkbox)

        # Region label display
        self.region_label: QLabel = QLabel("Hover over atlas to see region names")
        self.region_label.setMinimumWidth(400)
        self.region_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.control_layout.addWidget(self.region_label)

        self.control_layout.addStretch()  # Push controls to the left
        self.main_layout.addWidget(self.control_panel)

        self.atlas_view: GraphicsLayoutWidget = GraphicsLayoutWidget()
        self.atlas_viewbox: ViewBox = self.atlas_view.addViewBox()
        self.atlas_view.addLabel('Atlas View', row=0, col=0)
        self.transformed_view: GraphicsLayoutWidget = GraphicsLayoutWidget()
        self.brain_viewbox: ViewBox = self.transformed_view.addViewBox(row=1, col=0)
        self.transformed_view.addLabel('Brain Image', row=0, col=0)
        self.transformed_atlas_viewbox: ViewBox = self.transformed_view.addViewBox(row=1, col=1)
        self.transformed_view.addLabel('Transformed Atlas', row=0, col=1)

        self.main_layout.addWidget(self.atlas_view)
        self.main_layout.addWidget(self.transformed_view)

        self.slice_index: int = 0
        self.slice_data: Optional[VisualignSlice] = None
        self.image: Optional[np.ndarray] = None
        # Keep both color and grayscale copies of the loaded image
        self.color_image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.markers: Optional[np.ndarray] = None
        self.anchoring: Optional[np.ndarray] = None
        self.triangulation: Optional[Delaunay] = None
        self.atlas_item: Optional[ImageItem] = None
        self.transformed_image_item: Optional[ImageItem] = None
        self.original_line: Optional[PlotDataItem] = None
        self.transformed_line: Optional[PlotDataItem] = None

        self.raw_atlas_slice: Optional[npt.NDArray[np.int32]] = None
        self.display_slice: Optional[npt.NDArray[np.float64]] = None
        self.region_labels: dict = {}  # Maps region ID to region name

        self.drawing: bool = False
        self.line_points: List[Tuple[float, float]] = []

        # If no JSON file is provided, show the load button
        if json_file:
            self.load_data(json_file)
        else:
            self.load_button: QPushButton = QPushButton("Load VisuAlign JSON File")
            self.load_button.clicked.connect(self.load_data_interactively)
            self.main_layout.addWidget(self.load_button)

    def load_data(self, json_file: str) -> None:
        logging.info(f"Loading JSON file: {json_file}")
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            try:
                project = load_visualign_project(data)
            except ValidationError as ve:
                logging.error(f"Invalid JSON structure: {ve.message}")
                return

            # Load the first slice (TODO: Switch between slices)
            self.slice_data = project["slices"][0]
            self.markers = np.array(self.slice_data["markers"])
            self.anchoring = np.array(self.slice_data["anchoring"])

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

            # Keep both color (RGB) and grayscale versions; display based on checkbox
            self.color_image = np.fliplr(np.array(image.convert("RGB")))
            self.gray_image = np.fliplr(np.array(image.convert("L")))
            self.image = self.gray_image if self.grayscale_checkbox.isChecked() else self.color_image

            self.perform_triangulation()

            # use validated project for top-level fields
            self.load_atlas("cutlas/" + project["target"] + "/labels.nii.gz")
            self.load_region_labels("cutlas/" + project["target"] + "/labels.txt")

        except Exception as e:
            logging.error(f"Failed to load data: {e}")

    def load_data_interactively(self) -> None:
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", "JSON Files (*.json)"
        )
        if not json_path:
            logging.warning("No JSON file selected.")
            return

        self.load_data(json_path)

    def load_atlas(self, nifti_file: str) -> None:
        logging.info(f"Loading NIfTI file: {nifti_file}")
        try:
            self.atlas = Int32Slices(nifti_file)
            logging.info("NIfTI file loaded successfully.")

        except Exception as e:
            logging.error(f"Failed to load NIfTI file: {e}")

        self.draw_atlas()

    def draw_atlas(self) -> None:
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
        display_slice = np.zeros_like(atlas_slice, dtype=np.float64)

        # Get unique non-zero values (brain region IDs)
        unique_labels = np.unique(atlas_slice[atlas_slice > 0])
        logging.debug(f"Unique brain region labels: {len(unique_labels)} regions")

        # Map each unique label to a display value
        # TODO: Grab the colors from the labels.txt? - can use itk library to parse
        for i, label in enumerate(unique_labels):
            display_value = (i + 1) * (255.0 / len(unique_labels))
            display_slice[atlas_slice == label] = display_value

        self.display_slice = display_slice
        logging.debug(f"Display slice min/max: {display_slice.min()}/{display_slice.max()}")

        self.update_atlas_display()
        self.setup_mouse_tracking()
        self.display_transformed_views()

    def update_atlas_display(self) -> None:
        """Update the atlas display based on current settings."""
        if self.display_slice is None:
            return

        if self.outline_checkbox.isChecked():
            outline_slice = self.create_region_outlines()
            display_data = outline_slice
        else:
            display_data = self.display_slice

        logging.debug("Updating atlas display.")
        if self.atlas_item:
            self.atlas_viewbox.removeItem(self.atlas_item)
        
        # self.atlas_item = ImageItem(np.flipud(display_data.T))
        self.atlas_item = ImageItem(display_data.T)
        self.atlas_viewbox.addItem(self.atlas_item)
        self.atlas_viewbox.setAspectLocked(True)

    def create_region_outlines(self) -> npt.NDArray[np.int32]:
        """Create an outline-only version of the atlas slice."""
        if self.raw_atlas_slice is None:
            return np.zeros((1, 1), dtype=np.int32)

        outline_slice = np.zeros_like(self.raw_atlas_slice, dtype=np.int32)

        # Detect edges using numpy array operations
        # Based on: https://stackoverflow.com/a/29488679
        
        north = self.raw_atlas_slice[:-2, 1:-1]
        south = self.raw_atlas_slice[2:, 1:-1]
        west = self.raw_atlas_slice[1:-1, :-2]
        east = self.raw_atlas_slice[1:-1, 2:]
        
        center = self.raw_atlas_slice[1:-1, 1:-1]
        
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

    def toggle_outline_mode(self) -> None:
        """Toggle between filled regions and outline-only display."""
        self.update_atlas_display()

        if not hasattr(self, 'transformed_atlas_viewbox') or self.display_slice is None:
            return
        
        self.transformed_atlas_viewbox.clear()
        transformed_atlas = self.transform_atlas()
        # transformed_atlas_item = ImageItem(image=np.flipud(transformed_atlas.T))
        transformed_atlas_item = ImageItem(image=transformed_atlas.T)
        self.transformed_atlas_viewbox.addItem(transformed_atlas_item)

    def toggle_grayscale(self) -> None:
        """Toggle whether the brain image is shown in grayscale or color."""
        if self.color_image is None or self.gray_image is None:
            return
        self.image = self.gray_image if self.grayscale_checkbox.isChecked() else self.color_image
        # Refresh the views
        self.update_atlas_display()
        self.display_transformed_views()

    def display_transformed_views(self) -> None:
        """Display the brain image and transformed atlas in the bottom window."""
        if self.image is None or self.display_slice is None:
            return

        logging.debug("Displaying brain image and transformed atlas.")

        # Clear previous items to avoid stacking multiple items
        try:
            self.brain_viewbox.clear()
        except Exception:
            pass
        try:
            self.transformed_atlas_viewbox.clear()
        except Exception:
            pass

        if self.image.ndim == 2:
            brain_image_for_item = self.image.T
        elif self.image.ndim == 3: # colored image
            brain_image_for_item = np.transpose(self.image, (1, 0, 2))
        else:
            brain_image_for_item = self.image.T

        # Display brain image
        brain_image_item = ImageItem(image=brain_image_for_item)
        self.brain_viewbox.addItem(brain_image_item)
        self.brain_viewbox.setAspectLocked(True)

        # Create and display transformed atlas
        transformed_atlas = self.transform_atlas()
        transformed_atlas_item = ImageItem(image=transformed_atlas.T)
        self.transformed_atlas_viewbox.addItem(transformed_atlas_item)
        self.transformed_atlas_viewbox.setAspectLocked(True)

    def transform_atlas(self) -> npt.NDArray[np.float64]:
        """Transform the atlas using the triangulation to align with the brain image."""
        if self.display_slice is None or self.triangulation is None or self.image is None:
            return np.zeros((100, 100), dtype=np.float64)

        if self.outline_checkbox.isChecked():
            source_data = self.create_region_outlines()
        else:
            source_data = self.display_slice

        # Support color images: take height/width from first two dims
        output_height, output_width = self.image.shape[:2]
        transformed_atlas = np.zeros((output_height, output_width), dtype=np.float64)

        # Transform each pixel from atlas space to brain image space
        # TODO: optimize with vectorized operations instead of iteration
        for y in range(source_data.shape[0]):
            for x in range(source_data.shape[1]):
                # Ignore background pixels
                if source_data[y, x] == 0:
                    continue

                transformed_x, transformed_y = self.transform_point(x, y)

                tx, ty = int(round(transformed_x)), int(round(transformed_y))

                # Check bounds and assign pixel value
                if (0 <= tx < output_width and 0 <= ty < output_height):
                    transformed_atlas[ty, tx] = source_data[y, x]

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

    def setup_mouse_tracking(self) -> None:
        """Set up mouse tracking for the atlas view."""
        if self.atlas_item is None:
            return
        
        scene = self.atlas_view.scene()
        if scene:
            scene.sigMouseMoved.connect(self.on_mouse_move)  # type: ignore

    def on_mouse_move(self, pos: Any) -> None:
        """Handle mouse movement over the atlas."""
        if self.raw_atlas_slice is None or self.atlas_item is None:
            return

        # Convert scene coordinates to image coordinates
        if self.atlas_item.sceneBoundingRect().contains(pos):
            item_pos = self.atlas_item.mapFromScene(pos)
            x_disp, y_disp = int(item_pos.x()), int(item_pos.y())
            raw_h, raw_w = self.raw_atlas_slice.shape

            if (0 <= x_disp < raw_h and 0 <= y_disp < raw_w):
                # Get the region ID at this position
                region_id = self.raw_atlas_slice[y_disp, x_disp]

                if region_id > 0 and region_id in self.region_labels:
                    region_name = self.region_labels[region_id]
                    self.region_label.setText(f"Region: {region_name} (ID: {region_id})")
                else:
                    self.region_label.setText("Background region")
            else:
                self.region_label.setText("Hover over atlas to see region names")
        else:
            self.region_label.setText("Hover over atlas to see region names")

    def perform_triangulation(self) -> None:
        if self.markers is None:
            return
        # TODO: Should this use the original modified Delaunay from VisuAlign?
        points = self.markers[:, 2:4]
        self.triangulation = Delaunay(points)

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisuAlign Viewer")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--json", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = VisuAlignApp(json_file=args.json, debug=args.debug)
    window.show()
    sys.exit(app.exec_())
