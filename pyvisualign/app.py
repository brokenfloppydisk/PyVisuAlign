import sys
import json
import numpy as np
import logging
import argparse

from typing import Optional, Tuple, Dict
from enum import Enum
from pathlib import Path

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
    QLineEdit,
    QComboBox,
)
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QDoubleValidator, QIcon
from pyqtgraph import GraphicsLayoutWidget, ImageItem, ViewBox, PlotDataItem

from pyvisualign.project_data import (
    load_visualign_project, 
    VisualignProject,
    load_measurement_data,
    save_measurement_data,
    create_slice_measurements,
    MeasurementReferenceLine,
    MeasurementLine as MeasurementLineDict,
    SliceMeasurements,
    PyVisuAlignData
)
from pyvisualign.int32slices import Int32Slices
from pyvisualign.slice import Slice
from pyvisualign.measurement import Measurement

class MeasurementMode(Enum):
    NONE = 0
    REFERENCE = 1
    DRAW = 2
    ERASE = 3

class VisuAlignApp(QMainWindow):
    ASSETS_PATH = str(Path(__file__).resolve()) + "/../assets"
    def __init__(self, json_file: Optional[str] = None, debug: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("PyVisuAlign")
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
        
        opacity_grayscale_widget = QWidget()
        opacity_grayscale_layout = QVBoxLayout(opacity_grayscale_widget)
        opacity_grayscale_layout.setContentsMargins(0, 0, 0, 0)
        opacity_grayscale_layout.setSpacing(2)
        
        opacity_widget = QWidget()
        opacity_hlayout = QHBoxLayout(opacity_widget)
        opacity_hlayout.setContentsMargins(0, 0, 0, 0)
        opacity_hlayout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        opacity_label = QLabel("Atlas Opacity:")
        opacity_hlayout.addWidget(opacity_label)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(10)
        self.opacity_slider.setValue(5)
        self.opacity_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.opacity_slider.setTickInterval(1)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        opacity_hlayout.addWidget(self.opacity_slider, alignment=Qt.AlignmentFlag.AlignBottom)
        
        self.grayscale_checkbox: QCheckBox = QCheckBox("Grayscale Image")
        self.grayscale_checkbox.setChecked(False)
        self.grayscale_checkbox.stateChanged.connect(self.toggle_grayscale)
        
        opacity_grayscale_layout.addWidget(opacity_widget)
        opacity_grayscale_layout.addWidget(self.grayscale_checkbox)
        self.control_layout.addWidget(opacity_grayscale_widget)

        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)
        
        self.info_label: QLabel = QLabel("Select a JSON file to load")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info_label.setMinimumWidth(400)
        self.info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        info_layout.addWidget(self.info_label)
        
        self.atlas_info_label: QLabel = QLabel("Atlas: Unknown")
        self.atlas_info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.atlas_info_label.setMinimumWidth(400)
        self.atlas_info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        info_layout.addWidget(self.atlas_info_label)
        
        self.control_layout.addWidget(info_widget)
        self.control_layout.addStretch()
        
        # Measurement controls
        measurement_widget = QWidget()
        measurement_layout = QVBoxLayout(measurement_widget)
        measurement_layout.setContentsMargins(0, 0, 0, 0)
        measurement_layout.setSpacing(2)
        
        measurement_top_widget = QWidget()
        measurement_top_layout = QHBoxLayout(measurement_top_widget)
        measurement_top_layout.setContentsMargins(0, 0, 0, 0)
        measurement_top_layout.setSpacing(5)
        
        measurement_label = QLabel("Ref. Size:")
        measurement_top_layout.addWidget(measurement_label)
        
        self.reference_size_input = QLineEdit()
        self.reference_size_input.setText("1")
        self.reference_size_input.setMaximumWidth(80)
        self.reference_size_input.setValidator(QDoubleValidator(0.0, 1e9, 2))
        measurement_top_layout.addWidget(self.reference_size_input)
        
        self.unit_selector = QComboBox()
        self.unit_selector.addItems(["μm", "mm", "cm", "m", "px"])
        self.unit_selector.setCurrentIndex(1)
        self.unit_selector.setMaximumWidth(60)
        self.unit_selector.currentTextChanged.connect(self.on_unit_changed)
        measurement_top_layout.addWidget(self.unit_selector)
        
        measurement_bottom_widget = QWidget()
        measurement_bottom_layout = QHBoxLayout(measurement_bottom_widget)
        measurement_bottom_layout.setContentsMargins(0, 0, 0, 0)
        measurement_bottom_layout.setSpacing(5)
        
        self.set_reference_button = QPushButton('')
        self.set_reference_button.setIcon(QIcon(f'{self.ASSETS_PATH}/ruler.svg'))
        self.set_reference_button.setToolTip("Draw reference line")
        self.set_reference_button.setMaximumWidth(40)
        self.set_reference_button.setCheckable(True)
        self.set_reference_button.setEnabled(False)
        self.set_reference_button.clicked.connect(self.toggle_reference_mode)
        measurement_bottom_layout.addWidget(self.set_reference_button)
        
        self.draw_measure_button = QPushButton('')
        self.draw_measure_button.setIcon(QIcon(f'{self.ASSETS_PATH}/pencil.svg'))
        self.draw_measure_button.setToolTip("Draw measurement lines")
        self.draw_measure_button.setMaximumWidth(40)
        self.draw_measure_button.setCheckable(True)
        self.draw_measure_button.setEnabled(False)
        self.draw_measure_button.clicked.connect(self.toggle_draw_mode)
        measurement_bottom_layout.addWidget(self.draw_measure_button)
        
        self.erase_measure_button = QPushButton('')
        self.erase_measure_button.setIcon(QIcon(f'{self.ASSETS_PATH}/erase.svg'))
        self.erase_measure_button.setToolTip("Erase measurement lines")
        self.erase_measure_button.setMaximumWidth(40)
        self.erase_measure_button.setCheckable(True)
        self.erase_measure_button.setEnabled(False)
        self.erase_measure_button.clicked.connect(self.toggle_erase_mode)
        measurement_bottom_layout.addWidget(self.erase_measure_button)

        self.save_measurements_button = QPushButton('')
        self.save_measurements_button.setIcon(QIcon(f"{self.ASSETS_PATH}/download-circle.svg"))
        self.save_measurements_button.setToolTip("Save all measurements to file")
        self.save_measurements_button.setStyleSheet("background-color: white")
        self.save_measurements_button.setMaximumWidth(40)
        self.save_measurements_button.setCheckable(False)
        self.save_measurements_button.setEnabled(False)
        self.save_measurements_button.clicked.connect(self.save_measurements)
        measurement_bottom_layout.addWidget(self.save_measurements_button)
        
        measurement_layout.addWidget(measurement_top_widget)
        measurement_layout.addWidget(measurement_bottom_widget)
        self.control_layout.addWidget(measurement_widget)

        self.control_layout.addStretch()
        
        # Slice navigation controls
        slice_nav_widget = QWidget()
        slice_nav_layout = QVBoxLayout(slice_nav_widget)
        slice_nav_layout.setContentsMargins(0, 0, 0, 0)
        slice_nav_layout.setSpacing(2)

        self.slice_label: QLabel = QLabel("Slice: -")
        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        slice_buttons_widget = QWidget()
        slice_buttons_layout = QHBoxLayout(slice_buttons_widget)
        slice_buttons_layout.setContentsMargins(0, 0, 0, 0)
        slice_buttons_layout.setSpacing(5)
        
        self.prev_slice_button: QPushButton = QPushButton("< Previous Slice")
        self.prev_slice_button.clicked.connect(self.previous_slice)
        self.prev_slice_button.setEnabled(False)
        slice_buttons_layout.addWidget(self.prev_slice_button)
        
        self.next_slice_button: QPushButton = QPushButton("Next Slice >")
        self.next_slice_button.clicked.connect(self.next_slice)
        self.next_slice_button.setEnabled(False)
        slice_buttons_layout.addWidget(self.next_slice_button)
        
        slice_nav_layout.addWidget(self.slice_label)
        slice_nav_layout.addWidget(slice_buttons_widget)
        
        self.control_layout.addWidget(slice_nav_widget)
        
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
        self.slice_cache: Dict[int, Slice] = {}
        self.slice_cache_order: list = []
        self.max_cache_size: int = 20
        self.measurement_data: Optional[PyVisuAlignData] = None
        self.current_measurements: Dict[str, SliceMeasurements] = {}
        self.measurements_modified: bool = False
        
        self.measurement_mode: MeasurementMode = MeasurementMode.NONE
        self.line_start: Optional[Tuple[float, float]] = None
        self.temp_line_item: Optional[PlotDataItem] = None

        if json_file:
            self.load_data(json_file)
        else:
            self.load_button: QPushButton = QPushButton("Load VisuAlign JSON File")
            self.load_button.clicked.connect(self.load_data_interactively)
            self.main_layout.addWidget(self.load_button)

    def load_data(self, json_file: str) -> bool:
        """Load project data from a VisuAlign JSON file."""
        logging.info(f"Loading JSON file: {json_file}")
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            project = load_visualign_project(data)

            self.project_data = project
            self.json_file_path = json_file
            self.current_slice_index = 0
            
            self.load_atlas("cutlas/" + project["target"] + "/labels.nii.gz")
            self.load_region_labels("cutlas/" + project["target"] + "/labels.txt")

            if self.atlas is None or self.region_labels is None or self.color_map is None:
                logging.error("Atlas data not loaded")
                self.info_label.setText("Invalid File Loaded.")
                return False
            
            self.atlas_info_label.setText(f"Atlas: {project['target']}")
            
            self.measurement_data = load_measurement_data(json_file)
            if self.measurement_data:
                logging.info(f"Loaded measurement data for {len(self.measurement_data['measurements'])} slices")
                self.current_measurements = self.measurement_data['measurements'].copy()
            else:
                logging.info("No existing measurement data found")
                self.current_measurements = {}
            
            self.measurements_modified = False
            
            self.preload_slices()
            self.load_slice(self.current_slice_index)
            self.info_label.setText("JSON Successfully Loaded.")
            
            self.set_reference_button.setEnabled(True)
            self.erase_measure_button.setEnabled(True)
            self.save_measurements_button.setEnabled(False)
            return True
        except Exception as e:
            self.info_label.setText("Invalid File Loaded.")
            logging.error(f"Failed to load data: {e}")
            return False
    
    def load_data_interactively(self) -> None:
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", "JSON Files (*.json)"
        )
        if not json_path:
            logging.warning("No JSON file selected.")
            return

        if self.load_data(json_path):
            self.main_layout.removeWidget(self.load_button)
            self.load_button.deleteLater()
    
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
        
        self._load_slice_measurements()
        
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

    def _load_slice_measurements(self) -> None:
        """Load measurements from the central dictionary into the current slice."""
        if self.current_slice is None or self.project_data is None:
            return
        
        slice_filename = self.project_data["slices"][self.current_slice_index]["filename"]
        
        if slice_filename not in self.current_measurements:
            return
        
        slice_measurements = self.current_measurements[slice_filename]
        
        ref = slice_measurements.get('reference_line')
        if ref is not None:
            self.current_slice.set_reference_measurement(ref['size'], ref['unit'])
            start = tuple(ref['start'])
            end = tuple(ref['end'])
            self.current_slice.set_reference_line(
                (start[0], start[1]),
                (end[0], end[1])
            )
            self.draw_measure_button.setEnabled(True)
        else:
            self.draw_measure_button.setEnabled(False)
        
        for meas in slice_measurements.get('measurement_lines', []):
            start = tuple(meas['start'])
            end = tuple(meas['end'])
            self.current_slice.add_measurement_line(
                (start[0], start[1]),
                (end[0], end[1])
            )
    
    def _update_current_slice_measurements(self) -> None:
        """Update the central measurements dictionary with current slice's measurements."""
        if self.current_slice is None or self.project_data is None:
            return
        
        slice_filename = self.project_data["slices"][self.current_slice_index]["filename"]
        
        ref_line: Optional[MeasurementReferenceLine] = None
        if self.current_slice.reference_line is not None:
            ref_start, ref_end = self.current_slice.reference_line
            ref_line = {
                'start': [float(ref_start[0]), float(ref_start[1])],
                'end': [float(ref_end[0]), float(ref_end[1])],
                'size': float(self.current_slice.reference_size) if self.current_slice.reference_size else 0.0,
                'unit': self.current_slice.reference_unit,
                'pixel_distance': float(self.current_slice.calculate_pixel_distance(ref_start, ref_end))
            }
        
        meas_lines: list[MeasurementLineDict] = []
        for meas in self.current_slice.measurement_lines:
            meas_lines.append({
                'start': [float(meas.start[0]), float(meas.start[1])],
                'end': [float(meas.end[0]), float(meas.end[1])],
                'length': float(meas.length)
            })
        
        if ref_line or meas_lines:
            self.current_measurements[slice_filename] = create_slice_measurements(ref_line, meas_lines)
        elif slice_filename in self.current_measurements:
            del self.current_measurements[slice_filename]
        
        self.measurements_modified = True
        self.save_measurements_button.setIcon(QIcon(f"{self.ASSETS_PATH}/download-circle-solid.svg"))
        self.save_measurements_button.setStyleSheet("background-color: red")
        self.save_measurements_button.setEnabled(True)
    
    def save_measurements(self) -> None:
        """Save all measurements to the sidecar file in slice index order."""
        if self.json_file_path is None or self.project_data is None:
            return
        
        ordered_measurements: Dict[str, SliceMeasurements] = {}
        for slice_data in self.project_data["slices"]:
            slice_filename = slice_data["filename"]
            if slice_filename in self.current_measurements:
                ordered_measurements[slice_filename] = self.current_measurements[slice_filename]
        
        success = save_measurement_data(self.json_file_path, ordered_measurements)
        if success:
            self.measurements_modified = False
            self.save_measurements_button.setIcon(QIcon(f"{self.ASSETS_PATH}/download-circle.svg"))
            self.save_measurements_button.setStyleSheet("background-color: white")
            self.save_measurements_button.setEnabled(False)
            self.info_label.setText(f"Saved measurements for {len(ordered_measurements)} slices")
            logging.info(f"Saved measurements for {len(ordered_measurements)} slices")
        else:
            self.info_label.setText("Failed to save measurements")
            logging.error("Failed to save measurement data")

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

        # Start camera centered on image
        current_view_range = None
        current_view_range = {
            'xRange': self.brain_viewbox.viewRange()[0],
            'yRange': self.brain_viewbox.viewRange()[1]
        }

        self.brain_viewbox.clear()
        
        composite = np.transpose(self.current_slice.composite_image, (1, 0, 2))
        brain_image_item = ImageItem(image=composite)
        self.brain_viewbox.addItem(brain_image_item)
        self.brain_viewbox.setAspectLocked(True)
        
        if self.current_slice.reference_line is not None:
            ref_start, ref_end = self.current_slice.reference_line
            ref_line = PlotDataItem(
                [ref_start[0], ref_end[0]], 
                [ref_start[1], ref_end[1]],
                pen={'color': 'g', 'width': 3}
            )
            self.brain_viewbox.addItem(ref_line)
            
            if self.current_slice.reference_size is not None:
                mid_x = (ref_start[0] + ref_end[0]) / 2
                mid_y = (ref_start[1] + ref_end[1]) / 2
                from pyqtgraph import TextItem
                ref_label = TextItem(
                    f"{self.current_slice.reference_size:.1f} {self.current_slice.reference_unit}",
                    color='g',
                    anchor=(0.5, 0.5)
                )
                ref_label.setPos(mid_x, mid_y)
                self.brain_viewbox.addItem(ref_label)
        
        # Draw measurement lines
        for measurement in self.current_slice.measurement_lines:
            start = measurement.start
            end = measurement.end
            length = measurement.length
            
            measure_line = PlotDataItem(
                [start[0], end[0]], 
                [start[1], end[1]],
                pen={'color': 'r', 'width': 2}
            )
            self.brain_viewbox.addItem(measure_line)
            
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            from pyqtgraph import TextItem
            measure_label = TextItem(
                f"{length:.2f} {self.current_slice.reference_unit}",
                color='r',
                anchor=(0.5, 0.5)
            )
            measure_label.setPos(mid_x, mid_y)
            self.brain_viewbox.addItem(measure_label)
        
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
            scene.sigMouseClicked.connect(self.on_mouse_click)  # type: ignore
    
    def on_unit_changed(self, new_unit: str) -> None:
        """Handle changes to the measurement unit selector."""
        if self.current_slice is not None and self.current_slice.reference_line is not None:
            self.current_slice.reference_unit = new_unit
            self.display_brain()
    
    def toggle_reference_mode(self) -> None:
        """Toggle reference line drawing mode."""
        if self.set_reference_button.isChecked():
            self.measurement_mode = MeasurementMode.REFERENCE
            self.draw_measure_button.setChecked(False)
            self.erase_measure_button.setChecked(False)
            self.info_label.setText("Click two points to draw reference line")
        else:
            self.measurement_mode = MeasurementMode.NONE
            self.line_start = None
            self.clear_temp_line()
            self.info_label.setText("Hover over atlas to see region names")
    
    def toggle_draw_mode(self) -> None:
        """Toggle measurement line drawing mode."""
        if self.draw_measure_button.isChecked():
            self.measurement_mode = MeasurementMode.DRAW
            self.set_reference_button.setChecked(False)
            self.erase_measure_button.setChecked(False)
            self.info_label.setText("Click two points to draw measurement line")
        else:
            self.measurement_mode = MeasurementMode.NONE
            self.line_start = None
            self.clear_temp_line()
            self.info_label.setText("Hover over atlas to see region names")
    
    def toggle_erase_mode(self) -> None:
        """Toggle measurement line erasing mode."""
        if self.erase_measure_button.isChecked():
            self.measurement_mode = MeasurementMode.ERASE
            self.set_reference_button.setChecked(False)
            self.draw_measure_button.setChecked(False)
            self.info_label.setText("Click on a measurement line to erase it")
        else:
            self.measurement_mode = MeasurementMode.NONE
            self.info_label.setText("Hover over atlas to see region names")
    
    def clear_temp_line(self) -> None:
        """Clear the temporary line being drawn."""
        if self.temp_line_item is not None:
            self.brain_viewbox.removeItem(self.temp_line_item)
            self.temp_line_item = None
    
    def on_mouse_click(self, event) -> None:
        """Handle mouse clicks for measurement drawing."""
        if self.current_slice is None:
            return
        
        # Get the click position in view coordinates
        pos = event.scenePos()
        view_pos = self.brain_viewbox.mapSceneToView(pos)
        x, y = view_pos.x(), view_pos.y()
        
        if self.measurement_mode == MeasurementMode.REFERENCE:
            self.handle_reference_click(x, y)
        elif self.measurement_mode == MeasurementMode.DRAW:
            self.handle_draw_click(x, y)
        elif self.measurement_mode == MeasurementMode.ERASE:
            self.handle_erase_click(x, y)
    
    def handle_reference_click(self, x: float, y: float) -> None:
        """Handle clicks in reference line drawing mode."""
        if self.line_start is None:
            self.line_start = (x, y)
            self.info_label.setText("Click second point to complete reference line")
        else:
            try:
                ref_size = float(self.reference_size_input.text())
                if ref_size <= 0:
                    self.info_label.setText("Error: Reference size must be positive")
                    self.line_start = None
                    return
            except ValueError:
                self.info_label.setText("Error: Please enter a valid reference size")
                self.line_start = None
                return
            
            unit = self.unit_selector.currentText()
            
            if self.current_slice is not None:
                self.current_slice.set_reference_measurement(ref_size, unit)
                self.current_slice.set_reference_line(self.line_start, (x, y))
            
            self.line_start = None
            self.clear_temp_line()
            self.set_reference_button.setChecked(False)
            self.measurement_mode = MeasurementMode.NONE
            self.info_label.setText(f"Reference line set: {ref_size} {unit}")
            self.draw_measure_button.setEnabled(True)
            self.display_brain()
            self._update_current_slice_measurements()
    
    def handle_draw_click(self, x: float, y: float) -> None:
        """Handle clicks in measurement line drawing mode."""
        if self.current_slice is None:
            return
        
        if self.current_slice.reference_line is None:
            self.info_label.setText("Error: Please set a reference line first")
            return
        
        if self.line_start is None:
            self.line_start = (x, y)
            self.info_label.setText("Click second point to complete measurement line")
        else:
            self.current_slice.add_measurement_line(self.line_start, (x, y))
            self.line_start = None
            self.clear_temp_line()
            self.info_label.setText("Measurement line added")
            self.display_brain()
            self._update_current_slice_measurements()
    
    def handle_erase_click(self, x: float, y: float) -> None:
        """Handle clicks in erase mode."""
        if self.current_slice is None:
            return
        
        removed = self.current_slice.remove_measurement_line(x, y, threshold=10.0)
        
        if removed:
            self.info_label.setText("Measurement line removed")
            self.display_brain()
            self._update_current_slice_measurements()
        else:
            self.info_label.setText("No measurement line found at that location")

    def on_mouse_move(self, pos: QPointF) -> None:
        """Handle mouse movement over the composite image."""
        if (self.current_slice is None or 
            self.current_slice.transformed_slice is None or 
            self.region_labels is None):
            return

        view_pos = self.brain_viewbox.mapSceneToView(pos)
        x, y = view_pos.x(), view_pos.y()
        
        # Show temporary line while drawing
        if self.line_start is not None and self.measurement_mode in (MeasurementMode.REFERENCE, MeasurementMode.DRAW):
            self.clear_temp_line()
            line_x = [self.line_start[0], x]
            line_y = [self.line_start[1], y]
            self.temp_line_item = PlotDataItem(
                line_x, line_y,
                pen={'color': 'y', 'width': 2, 'style': Qt.PenStyle.DashLine}
            )
            self.brain_viewbox.addItem(self.temp_line_item)
        
        if self.measurement_mode != MeasurementMode.NONE:
            return
        
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
                self.info_label.setText(f"Region: {region_name} (ID: {region_id})")
            else:
                self.info_label.setText("Background region")
        else:
            self.info_label.setText("Hover over atlas to see region names")

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