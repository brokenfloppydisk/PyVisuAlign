import sys
import json
import numpy as np
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
    QSlider,
)
from PyQt5.QtCore import QPointF, Qt
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ViewBox

from pyvisualign.project_data import load_visualign_project, VisualignProject
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

        self.central_widget: QWidget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout: QVBoxLayout = QVBoxLayout(self.central_widget)

        self.control_panel: QWidget = QWidget()
        self.control_layout: QHBoxLayout = QHBoxLayout(self.control_panel)
        
        opacity_label = QLabel("Atlas Opacity:")
        self.control_layout.addWidget(opacity_label)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(10)
        self.opacity_slider.setValue(5)
        self.opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.opacity_slider.setTickInterval(1)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        self.control_layout.addWidget(self.opacity_slider)
        
        self.grayscale_checkbox: QCheckBox = QCheckBox("Grayscale Image")
        self.grayscale_checkbox.setChecked(False)
        self.grayscale_checkbox.stateChanged.connect(self.toggle_grayscale)
        self.control_layout.addWidget(self.grayscale_checkbox)

        self.region_label: QLabel = QLabel("Hover over atlas to see region names")
        self.region_label.setMinimumWidth(400)
        self.region_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.control_layout.addWidget(self.region_label)

        self.control_layout.addStretch()
        
        self.prev_slice_button: QPushButton = QPushButton("◀ Previous Slice")
        self.prev_slice_button.clicked.connect(self.previous_slice)
        self.prev_slice_button.setEnabled(False)
        self.control_layout.addWidget(self.prev_slice_button)
        
        self.slice_label: QLabel = QLabel("Slice: -")
        self.control_layout.addWidget(self.slice_label)
        
        self.next_slice_button: QPushButton = QPushButton("Next Slice ▶")
        self.next_slice_button.clicked.connect(self.next_slice)
        self.next_slice_button.setEnabled(False)
        self.control_layout.addWidget(self.next_slice_button)
        
        self.main_layout.addWidget(self.control_panel)

        self.image_layout: GraphicsLayoutWidget = GraphicsLayoutWidget()
        self.brain_viewbox: ViewBox = self.image_layout.addViewBox(row=1, col=0)
        self.image_layout.addLabel('Brain Image with Atlas Overlay', row=0, col=0)

        self.main_layout.addWidget(self.image_layout)

        self.atlas: Optional[Int32Slices] = None
        self.region_labels: Optional[dict] = None
        self.color_map: Optional[dict] = None
        self.current_slice: Optional[Slice] = None
        self.first_display: bool = True
        self.use_grayscale: bool = False
        self.project_data: Optional[VisualignProject] = None
        self.current_slice_index: int = 0
        self.json_file_path: Optional[str] = None
        self.slice_cache: dict = {}
        self.slice_cache_order: list = []
        self.max_cache_size: int = 20

        if json_file:
            self.load_data(json_file)
        else:
            self.load_button: QPushButton = QPushButton("Load VisuAlign JSON File")
            self.load_button.clicked.connect(self.load_data_interactively)
            self.main_layout.addWidget(self.load_button)

    def load_data(self, json_file: str) -> None:
        """Load project data from a VisuAlign JSON file."""
        logging.info(f"Loading JSON file: {json_file}")
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            try:
                project = load_visualign_project(data)
            except ValidationError as ve:
                logging.error(f"Invalid JSON structure: {ve.message}")
                return

            self.project_data = project
            self.json_file_path = json_file
            self.current_slice_index = 0
            
            self.load_atlas("cutlas/" + project["target"] + "/labels.nii.gz")
            self.load_region_labels("cutlas/" + project["target"] + "/labels.txt")

            if self.atlas is None or self.region_labels is None or self.color_map is None:
                logging.error("Atlas data not loaded")
                return
            
            self.preload_slices()
            self.load_slice(self.current_slice_index)

        except Exception as e:
            logging.error(f"Failed to load data: {e}")
    
    def preload_slices(self) -> None:
        """Preload up to 5 slices at startup for faster navigation."""
        if self.project_data is None or self.json_file_path is None:
            return
        if self.atlas is None or self.region_labels is None or self.color_map is None:
            return
        
        total_slices = len(self.project_data["slices"])
        slices_to_load = min(total_slices, 5)
        
        logging.info(f"Preloading {slices_to_load} slices...")
        
        for i in range(slices_to_load):
            slice_data = self.project_data["slices"][i]
            slice_obj = Slice(slice_data, self.atlas, self.region_labels, self.color_map, self.json_file_path)
            slice_obj.generate_slice()
            self.slice_cache[i] = slice_obj
            self.slice_cache_order.append(i)
        
        logging.info(f"Preloading complete: {len(self.slice_cache)} slices cached")
    
    
    def load_slice(self, slice_index: int) -> None:
        """Load and display a specific slice by index."""
        if self.project_data is None or self.json_file_path is None:
            logging.error("Project data not loaded")
            return
        if slice_index < 0 or slice_index >= len(self.project_data["slices"]):
            logging.error(f"Invalid slice index: {slice_index}")
            return
        if self.atlas is None or self.region_labels is None or self.color_map is None:
            return
        
        self.current_slice_index = slice_index
        
        if slice_index in self.slice_cache:
            logging.debug(f"Loading slice {slice_index} from cache")
            self.current_slice = self.slice_cache[slice_index]
            if slice_index in self.slice_cache_order:
                self.slice_cache_order.remove(slice_index)
            self.slice_cache_order.append(slice_index)
        else:
            logging.debug(f"Creating new slice {slice_index}")
            slice_data = self.project_data["slices"][slice_index]
            self.current_slice = Slice(slice_data, self.atlas, self.region_labels, self.color_map, self.json_file_path)
            self.current_slice.generate_slice()
            self.slice_cache[slice_index] = self.current_slice
            self.slice_cache_order.append(slice_index)
            
            if len(self.slice_cache) > self.max_cache_size:
                oldest_index = self.slice_cache_order.pop(0)
                del self.slice_cache[oldest_index]
                logging.debug(f"Evicted slice {oldest_index} from cache")
        
        if self.current_slice is None:
            return
        
        self.update_slice_navigation()
        self.display_brain()
        if not hasattr(self, '_mouse_tracking_setup'):
            self.setup_mouse_tracking()
            self._mouse_tracking_setup = True

    def update_slice_navigation(self) -> None:
        """Update the slice navigation buttons and label."""
        if self.project_data is None:
            return
        
        total_slices = len(self.project_data["slices"])
        self.slice_label.setText(f"Slice: {self.current_slice_index + 1} / {total_slices}")
        
        # Enable/disable buttons based on position
        self.prev_slice_button.setEnabled(self.current_slice_index > 0)
        self.next_slice_button.setEnabled(self.current_slice_index < total_slices - 1)
    
    def previous_slice(self) -> None:
        """Navigate to the previous slice."""
        if self.current_slice_index > 0:
            self.load_slice(self.current_slice_index - 1)
    
    def next_slice(self) -> None:
        """Navigate to the next slice."""
        if self.project_data is not None and self.current_slice_index < len(self.project_data["slices"]) - 1:
            self.load_slice(self.current_slice_index + 1)

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
        """Load brain region labels and colors from the labels.txt file."""
        logging.info(f"Loading region labels and colors from: {labels_file}")
        try:
            self.region_labels = {}
            self.color_map = {}

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
                    r = int(parts[1])
                    g = int(parts[2])
                    b = int(parts[3])
                    label = parts[7].strip('"')
                    
                    self.region_labels[region_id] = label
                    self.color_map[region_id] = (r, g, b)

            logging.info(f"Loaded {len(self.region_labels)} region labels with colors")

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
            if self.current_slice.colored_transformed_slice is None:
                return
            display_slice = ImageItem(image=self.current_slice.colored_transformed_slice.T)
        
        if display_slice:
            self.atlas_viewbox.clear()
            self.atlas_viewbox.addItem(display_slice)
        else:
            logging.warning("Outline could not be toggled.")

    def toggle_grayscale(self) -> None:
        """Toggle whether the brain image is shown in grayscale or color."""
        if self.current_slice is None:
            return
        
        self.use_grayscale = self.grayscale_checkbox.isChecked()
        
        base_image = self.current_slice.gray_image if self.use_grayscale else self.current_slice.color_image
        
        opacity = self.opacity_slider.value() / 10.0  # Convert 0-10 to 0.0-1.0
        self.current_slice.create_composite_image(opacity, base_image=base_image)
        self.display_brain()

    def update_opacity(self) -> None:
        """Update the opacity of the atlas overlay"""
        if self.current_slice is None:
            return
        
        base_image = self.current_slice.gray_image if self.use_grayscale else self.current_slice.color_image
            
        opacity = self.opacity_slider.value() / 10.0  # Convert 0-10 to 0.0-1.0
        self.current_slice.create_composite_image(opacity, base_image=base_image)
        self.display_brain()

    def display_brain(self) -> None:
        """Display the brain image with atlas overlay in the viewer."""
        if self.current_slice is None or self.current_slice.composite_image is None:
            return

        logging.debug("Displaying composite image")

        current_view_range = None
        try:
            current_view_range = {
                'xRange': self.brain_viewbox.viewRange()[0],
                'yRange': self.brain_viewbox.viewRange()[1]
            }
        except Exception:
            pass

        try:
            self.brain_viewbox.clear()
        except Exception:
            pass
        
        composite = np.transpose(self.current_slice.composite_image, (1, 0, 2))
        brain_image_item = ImageItem(image=composite)
        self.brain_viewbox.addItem(brain_image_item)
        self.brain_viewbox.setAspectLocked(True)
        
        if self.first_display:
            self.brain_viewbox.autoRange(padding=0.02)
            self.first_display = False
        elif current_view_range is not None:
            self.brain_viewbox.setRange(
                xRange=current_view_range['xRange'],
                yRange=current_view_range['yRange'],
                padding=0
            )

    def setup_mouse_tracking(self) -> None:
        """Set up mouse tracking for the brain view."""
        scene = self.brain_viewbox.scene()
        if scene:
            scene.sigMouseMoved.connect(self.on_mouse_move)  # type: ignore

    def on_mouse_move(self, pos: QPointF) -> None:
        """Handle mouse movement over the composite image."""
        if (self.current_slice is None or 
            self.current_slice.transformed_slice is None or 
            self.region_labels is None):
            return

        view_pos = self.brain_viewbox.mapSceneToView(pos)
        
        # Get atlas data with region IDs (not the RGB version)
        if hasattr(self.current_slice, 'scaled_transformed_slice') and self.current_slice.scaled_transformed_slice is not None:
            transformed_slice = self.current_slice.scaled_transformed_slice
        else:
            transformed_slice = self.current_slice.transformed_slice

        x_disp = int(view_pos.y())  # x and y are swapped due to transpose
        y_disp = int(view_pos.x())
        
        h, w = transformed_slice.shape
        if (0 <= x_disp < h and 0 <= y_disp < w):
            region_id = transformed_slice[x_disp, y_disp]
            if region_id > 0 and region_id in self.region_labels:
                region_name = self.region_labels[region_id]
                self.region_label.setText(f"Region: {region_name} (ID: {region_id})")
            else:
                self.region_label.setText("Background region")
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