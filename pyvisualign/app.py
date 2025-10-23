import sys
import json
import numpy as np
import numpy.typing as npt
import logging
import argparse
from typing import Optional
from jsonschema import ValidationError
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
from PyQt5.QtCore import QPointF
from pyqtgraph import GraphicsLayoutWidget, GraphicsScene, ImageItem, PlotDataItem, ViewBox
from typing import Optional

from pyvisualign.project_data import load_visualign_project, VisualignSlice
from pyvisualign.int32slices import Int32Slices
from pyvisualign.slice import Slice


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

        self.image_layout: GraphicsLayoutWidget = GraphicsLayoutWidget()
        self.brain_viewbox: ViewBox = self.image_layout.addViewBox(row=1, col=0)
        self.image_layout.addLabel('Brain Image', row=0, col=0)
        self.atlas_viewbox: ViewBox = self.image_layout.addViewBox(row=1, col=1)
        self.image_layout.addLabel('Transformed Atlas', row=0, col=1)
        self.atlas_image: Optional[ImageItem] = None

        self.main_layout.addWidget(self.image_layout)

        self.atlas: Optional[Int32Slices] = None
        self.region_labels: Optional[dict] = None
        self.current_slice: Optional[Slice] = None
        self.current_image: Optional[npt.NDArray] = None

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

            self.load_atlas("cutlas/" + project["target"] + "/labels.nii.gz")
            self.load_region_labels("cutlas/" + project["target"] + "/labels.txt")

            if self.atlas is None or self.region_labels is None:
                logging.error("Atlas data not loaded")
                return
            
            self.current_slice = Slice(self.slice_data, self.atlas, self.region_labels, json_file)
            self.current_slice.generate_slice()
            self.current_image = self.current_slice.gray_image if self.grayscale_checkbox.isChecked() else self.current_slice.color_image

            self.display_brain()
            self.display_atlas()
            self.setup_mouse_tracking()

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
    
    def load_region_labels(self, labels_file: str) -> None:
        """Load brain region labels from the labels.txt file."""
        logging.info(f"Loading region labels from: {labels_file}")
        try:
            self.region_labels = {}

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

    def toggle_outline_mode(self) -> None:
        """Toggle between filled regions and outline-only display."""

        if not hasattr(self, 'transformed_atlas_viewbox') or self.current_slice is None:
            return
        
        display_slice = None

        if self.outline_checkbox.isChecked():
            if self.current_slice.wireframe_slice is None:
                return
            display_slice = ImageItem(image=self.current_slice.wireframe_slice.T)
        else:
            if self.current_slice.normalized_transformed_slice is None:
                return
            display_slice = ImageItem(image=self.current_slice.normalized_transformed_slice.T)
        
        if display_slice:
            self.atlas_viewbox.clear()
            self.atlas_viewbox.addItem(display_slice)
        else:
            logging.warning("Outline could not be toggled.")

    def toggle_grayscale(self) -> None:
        """Toggle whether the brain image is shown in grayscale or color."""
        if self.current_slice is None:
            return
        
        self.current_image = self.current_slice.gray_image if self.grayscale_checkbox.isChecked() else self.current_slice.color_image
        self.display_brain()

    def display_brain(self) -> None:
        if self.current_image is None:
            return

        logging.debug("Displaying brain image.")

        # Clear previous items to avoid stacking multiple items
        try:
            self.brain_viewbox.clear()
        except Exception:
            pass

        if self.current_image.ndim == 3: # colored image
            brain_image = np.transpose(self.current_image, (1, 0, 2))
        else:
            brain_image = self.current_image.T

        # Display brain image
        brain_image_item = ImageItem(image=brain_image)
        self.brain_viewbox.addItem(brain_image_item)
        self.brain_viewbox.setAspectLocked(True)

    def display_atlas(self) -> None:
        """Display the brain image and transformed atlas."""
        if self.current_slice is None or self.current_slice.normalized_transformed_slice is None:
            return

        logging.debug("Displaying transformed atlas")

        # Clear previous items to avoid stacking multiple items
        try:
            self.atlas_viewbox.clear()
        except Exception:
            pass

        # Display transformed atlas
        self.atlas_image = ImageItem(image=self.current_slice.normalized_transformed_slice.T)
        self.atlas_viewbox.addItem(self.atlas_image)
        self.atlas_viewbox.setAspectLocked(True)

    def setup_mouse_tracking(self) -> None:
        """Set up mouse tracking for the atlas view."""
        if self.atlas_image is None:
            return
        
        scene = self.atlas_viewbox.scene()
        if scene:
            scene.sigMouseMoved.connect(self.on_mouse_move)  # type: ignore

    def on_mouse_move(self, pos: QPointF) -> None:
        """Handle mouse movement over the atlas."""
        if self.atlas_image is None or self.region_labels is None:
            return
        if self.current_slice is None or self.current_slice.transformed_slice is None:
            logging.warning("Missing atlas data!")
            return

        # Convert scene coordinates to image coordinates
        if self.atlas_image.sceneBoundingRect().contains(pos):
            item_pos = self.atlas_image.mapFromScene(pos)
            x_disp, y_disp = int(item_pos.x()), int(item_pos.y())
            raw_h, raw_w = self.current_slice.transformed_slice.shape

            if (0 <= x_disp < raw_w and 0 <= y_disp < raw_h):
                # Get the region ID at this position
                region_id = self.current_slice.transformed_slice[y_disp, x_disp]

                if region_id > 0 and region_id in self.region_labels:
                    region_name = self.region_labels[region_id]
                    self.region_label.setText(f"Region: {region_name} (ID: {region_id})")
                else:
                    self.region_label.setText("Background region")
            else:
                self.region_label.setText("Hover over atlas to see region names")
        else:
            self.region_label.setText("Hover over atlas to see region names")

def run():
    parser = argparse.ArgumentParser(description="VisuAlign Viewer")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--json", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = VisuAlignApp(json_file=args.json, debug=args.debug)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()