"""
Module - NIfTI-1 Dataset Handling

Heavily based on the Java version at https://github.com/NIFTI-Imaging/nifti_java
"""

import gzip
import numpy as np
from typing import Optional, Tuple
from io import BytesIO
import sys
from pathlib import Path
from nifticonstants import *
from nifti.niftiio import EndianCorrectReader


class Nifti1Dataset:
    """
    Python port of the Java Nifti1Dataset class for reading/writing NIfTI-1 datasets.

    This class provides an API for reading and writing NIfTI-1 format neuroimaging data files.
    It handles both single-file (.nii) and dual-file (.hdr/.img) formats, with optional
    gzip compression support.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new NIfTI-1 dataset.

        Args:
            name: Optional filename for an existing dataset.
        """
        self._set_defaults()
        if name:
            self._check_name(name)

    def _set_defaults(self):
        """Set default values for all NIfTI-1 header fields"""
        # File properties
        self.ds_hdrname = ""
        self.ds_datname = ""
        self.ds_is_nii = False
        self.big_endian = sys.byteorder == "big"

        # Derived dimensions
        self.XDIM = 0
        self.YDIM = 0
        self.ZDIM = 0
        self.TDIM = 0
        self.DIM5 = 0
        self.DIM6 = 0
        self.DIM7 = 0

        # Unpacked fields
        self.freq_dim = 0
        self.phase_dim = 0
        self.slice_dim = 0
        self.xyz_unit_code = NiftiUnit.NIFTI_UNITS_UNKNOWN
        self.t_unit_code = NiftiUnit.NIFTI_UNITS_UNKNOWN
        self.qfac = 1

        # Extensions
        self.extensions_list = []
        self.extension_blobs = []

        # Header fields
        self.sizeof_hdr = NiftiConstants.ANZ_HDR_SIZE
        self.data_type_string = "\x00" * 10
        self.db_name = "\x00" * 18
        self.extents = 0
        self.session_error = 0
        self.regular = "\x00"
        self.dim_info = "\x00"
        self.dim = [0] * 8
        self.intent = [0.0] * 3
        self.intent_code = NiftiIntent.NIFTI_INTENT_NONE
        self.datatype = NiftiDataType.DT_NONE
        self.bitpix = 0
        self.slice_start = 0
        self.pixdim = [1.0] + [0.0] * 7
        self.vox_offset = 0.0
        self.scl_slope = 0.0
        self.scl_inter = 0.0
        self.slice_end = 0
        self.slice_code = NiftiSliceOrder.NIFTI_SLICE_UNKNOWN
        self.xyzt_units = 0
        self.cal_max = 0.0
        self.cal_min = 0.0
        self.slice_duration = 0.0
        self.toffset = 0.0
        self.glmax = 0
        self.glmin = 0
        self.descrip = "\x00" * 80
        self.aux_file = "\x00" * 24
        self.qform_code = NiftiXformCode.NIFTI_XFORM_UNKNOWN
        self.sform_code = NiftiXformCode.NIFTI_XFORM_UNKNOWN
        self.quatern = [0.0] * 3
        self.qoffset = [0.0] * 3
        self.srow_x = [0.0] * 4
        self.srow_y = [0.0] * 4
        self.srow_z = [0.0] * 4
        self.intent_name = "\x00" * 16
        self.magic = NiftiConstants.NII_MAGIC_STRING
        self.extension = [0, 0, 0, 0]

    def _check_name(self, name: str):
        """
        Check input filename and determine actual disk filenames for header and data.

        Args:
            name: Input filename that may or may not include extensions
        """
        wname = name

        # Strip .gz suffix - we'll determine compression from actual files
        if wname.endswith(NiftiConstants.GZIP_EXT):
            wname = wname[: -len(NiftiConstants.GZIP_EXT)]

        # If no extension specified, look for existing files
        if not any(
            wname.endswith(ext)
            for ext in [
                NiftiConstants.ANZ_HDR_EXT,
                NiftiConstants.ANZ_DAT_EXT,
                NiftiConstants.NII_EXT,
            ]
        ):
            # Check for .nii files first
            if Path(wname + NiftiConstants.NII_EXT).exists():
                wname += NiftiConstants.NII_EXT
            elif Path(
                wname + NiftiConstants.NII_EXT + NiftiConstants.GZIP_EXT
            ).exists():
                wname += NiftiConstants.NII_EXT
            elif Path(wname + NiftiConstants.ANZ_HDR_EXT).exists():
                wname += NiftiConstants.ANZ_HDR_EXT
            elif Path(
                wname + NiftiConstants.ANZ_HDR_EXT + NiftiConstants.GZIP_EXT
            ).exists():
                wname += NiftiConstants.ANZ_HDR_EXT
            elif Path(wname + NiftiConstants.ANZ_DAT_EXT).exists():
                wname += NiftiConstants.ANZ_HDR_EXT  # Use .hdr for .img files
            elif Path(
                wname + NiftiConstants.ANZ_DAT_EXT + NiftiConstants.GZIP_EXT
            ).exists():
                wname += NiftiConstants.ANZ_HDR_EXT  # Use .hdr for .img files

        # Set filenames based on extension
        if wname.endswith(NiftiConstants.ANZ_HDR_EXT):
            self.ds_hdrname = wname
            self.ds_datname = (
                wname[: -len(NiftiConstants.ANZ_HDR_EXT)] + NiftiConstants.ANZ_DAT_EXT
            )
        elif wname.endswith(NiftiConstants.ANZ_DAT_EXT):
            self.ds_datname = wname
            self.ds_hdrname = (
                wname[: -len(NiftiConstants.ANZ_DAT_EXT)] + NiftiConstants.ANZ_HDR_EXT
            )
        elif wname.endswith(NiftiConstants.NII_EXT):
            self.ds_hdrname = wname
            self.ds_datname = wname
            self.ds_is_nii = True

        # Check for gzipped versions
        if Path(self.ds_hdrname + NiftiConstants.GZIP_EXT).exists():
            self.ds_hdrname += NiftiConstants.GZIP_EXT
        if Path(self.ds_datname + NiftiConstants.GZIP_EXT).exists():
            self.ds_datname += NiftiConstants.GZIP_EXT

    def _unpack_dim_info(self, dim_info_byte: int) -> Tuple[int, int, int]:
        """Unpack the dim_info byte into freq_dim, phase_dim, slice_dim"""
        freq_dim = dim_info_byte & 3
        phase_dim = (dim_info_byte >> 2) & 3
        slice_dim = (dim_info_byte >> 4) & 3
        return freq_dim, phase_dim, slice_dim

    def _pack_dim_info(self, freq: int, phase: int, slice: int) -> int:
        """Pack freq_dim, phase_dim, slice_dim into a single byte"""
        return (slice & 3) << 4 | (phase & 3) << 2 | (freq & 3)

    def _unpack_units(self, units_byte: int) -> Tuple[NiftiUnit, NiftiUnit]:
        """Unpack xyzt_units into xyz and time unit codes"""
        xyz_units = NiftiUnit(units_byte & 0x07) # bits 0,1,2
        t_units = NiftiUnit(units_byte & 0x38) # bits 3,4,5
        return xyz_units, t_units

    def _pack_units(self, xyz: int, t: int) -> int:
        """Pack xyz and time unit codes into xyzt_units byte"""
        return (xyz & 0x07) | (t & 0x38)

    def exists(self) -> bool:
        """Check if both header and data files exist"""
        return self.exists_hdr() and self.exists_dat()

    def exists_hdr(self) -> bool:
        """Check if header file exists"""
        return Path(self.ds_hdrname).exists()

    def exists_dat(self) -> bool:
        """Check if data file exists"""
        return Path(self.ds_datname).exists()

    def read_header(self):
        """Read header information from disk into memory"""
        # First pass: determine endianness by reading dim[0]
        reader = EndianCorrectReader(
            self.ds_hdrname, True
        )  # Assume big endian initially
        reader.skip(40)  # Skip to dim[0]
        dim0 = reader.read_int16()
        reader.close()

        # Check if dim[0] makes sense for big endian
        self.big_endian = 1 <= dim0 <= 7

        # Second pass: read the full header with correct endianness
        reader = EndianCorrectReader(self.ds_hdrname, self.big_endian)

        # Read header fields in order
        self.sizeof_hdr = reader.read_int32()
        self.data_type_string = reader.read_string(10)
        self.db_name = reader.read_string(18)
        self.extents = reader.read_int32()
        self.session_error = reader.read_int16()
        self.regular = chr(reader.read_uint8())

        dim_info_byte = reader.read_uint8()
        self.dim_info = chr(dim_info_byte)
        self.freq_dim, self.phase_dim, self.slice_dim = self._unpack_dim_info(
            dim_info_byte
        )

        # Read dimensions
        for i in range(8):
            self.dim[i] = reader.read_int16()

        # Set convenience dimension variables
        if self.dim[0] > 0:
            self.XDIM = self.dim[1]
        if self.dim[0] > 1:
            self.YDIM = self.dim[2]
        if self.dim[0] > 2:
            self.ZDIM = self.dim[3]
        if self.dim[0] > 3:
            self.TDIM = self.dim[4]
        if self.dim[0] > 4:
            self.DIM5 = self.dim[5]
        if self.dim[0] > 5:
            self.DIM6 = self.dim[6]
        if self.dim[0] > 6:
            self.DIM7 = self.dim[7]

        # Read intent parameters
        for i in range(3):
            self.intent[i] = reader.read_float32()

        self.intent_code = NiftiIntent(reader.read_int16())
        self.datatype = NiftiDataType(reader.read_int16())
        self.bitpix = reader.read_int16()
        self.slice_start = reader.read_int16()

        # Read pixel dimensions
        for i in range(8):
            self.pixdim[i] = reader.read_float32()
        self.qfac = int(self.pixdim[0])

        self.vox_offset = reader.read_float32()
        self.scl_slope = reader.read_float32()
        self.scl_inter = reader.read_float32()
        self.slice_end = reader.read_int16()
        self.slice_code = NiftiSliceOrder(reader.read_uint8())

        xyzt_units_byte = reader.read_uint8()
        self.xyzt_units = xyzt_units_byte
        self.xyz_unit_code, self.t_unit_code = self._unpack_units(xyzt_units_byte)

        self.cal_max = reader.read_float32()
        self.cal_min = reader.read_float32()
        self.slice_duration = reader.read_float32()
        self.toffset = reader.read_float32()
        self.glmax = reader.read_int32()
        self.glmin = reader.read_int32()

        self.descrip = reader.read_string(80)
        self.aux_file = reader.read_string(24)

        self.qform_code = NiftiXformCode(reader.read_int16())
        self.sform_code = NiftiXformCode(reader.read_int16())

        # Read quaternion parameters
        for i in range(3):
            self.quatern[i] = reader.read_float32()
        for i in range(3):
            self.qoffset[i] = reader.read_float32()

        # Read affine transform rows
        for i in range(4):
            self.srow_x[i] = reader.read_float32()
        for i in range(4):
            self.srow_y[i] = reader.read_float32()
        for i in range(4):
            self.srow_z[i] = reader.read_float32()

        self.intent_name = reader.read_string(16)
        self.magic = reader.read_string(4)

        # Read extension header if present
        if self.ds_is_nii:
            self._read_nii_extensions(reader)
        else:
            self._read_np1_extensions(reader)

        reader.close()

    def _read_nii_extensions(self, reader):
        """Read extensions from .nii file"""
        # Read extension bytes
        for i in range(4):
            self.extension[i] = reader.read_uint8()

        if self.extension[0] != 0:
            start_addr = NiftiConstants.ANZ_HDR_SIZE + 4

            while start_addr < int(self.vox_offset):
                # Read extension size and code
                size = reader.read_int32()
                code = reader.read_int32()

                # Read extension data
                data = reader.read(size - NiftiConstants.EXT_KEY_SIZE)

                self.extensions_list.append((size, code))
                self.extension_blobs.append(data)

                start_addr += size

    def _read_np1_extensions(self, reader):
        """Read extensions from .hdr/.img file"""
        try:
            # Read extension bytes
            for i in range(4):
                self.extension[i] = reader.read_uint8()

            if self.extension[0] != 0:
                while True:
                    try:
                        # Read extension size and code
                        size = reader.read_int32()
                        code = reader.read_int32()

                        # Read extension data
                        data = reader.read(size - NiftiConstants.EXT_KEY_SIZE)

                        self.extensions_list.append((size, code))
                        self.extension_blobs.append(data)

                    except:
                        # End of file reached
                        break
        except:
            # Extension bytes not present - this is OK for .hdr files
            pass

    def print_header(self):
        """Print header information to stdout"""
        print(f"\nDataset header file:\t\t\t\t{self.ds_hdrname}")
        print(f"Dataset data file:\t\t\t\t{self.ds_datname}")
        print(f"Size of header:\t\t\t\t\t{self.sizeof_hdr}")
        print(f"File offset to data blob:\t\t\t{self.vox_offset}")
        print(f"Endianness:\t\t\t\t\t{'big' if self.big_endian else 'little'}")
        print(f"Magic filetype string:\t\t\t\t{repr(self.magic)}")

        # Dataset info
        print(
            f"Datatype:\t\t\t\t\t{self.datatype} ({self.datatype.name})"
        )
        print(f"Bits per voxel:\t\t\t\t\t{self.bitpix}")
        print(f"Scaling slope and intercept:\t\t\t{self.scl_slope} {self.scl_inter}")

        # Dimensions
        dim_str = " ".join(str(self.dim[i]) for i in range(self.dim[0] + 1))
        print(f"Dataset dimensions (Count, X,Y,Z,T...):\t\t{dim_str}")

        pixdim_str = " ".join(str(self.pixdim[i]) for i in range(1, self.dim[0] + 1))
        print(f"Grid spacings (X,Y,Z,T,...):\t\t\t{pixdim_str}")

        print(f"XYZ units:\t\t\t\t\t{self.xyz_unit_code} ({self.xyz_unit_code.name})")
        print(f"T units:\t\t\t\t\t{self.t_unit_code} ({self.t_unit_code.name})")
        print(f"T offset:\t\t\t\t\t{self.toffset}")

        # Intent
        intent_str = " ".join(str(self.intent[i]) for i in range(3))
        print(f"Intent parameters:\t\t\t\t{intent_str}")
        print(f"Intent code:\t\t\t\t\t{self.intent_code} ({self.intent_code.name})")

        print(f"Cal. (display) max/min:\t\t\t\t{self.cal_max} {self.cal_min}")

        # Slice timing
        print(f"Slice timing code:\t\t\t\t{self.slice_code} ({self.slice_code.name})")
        print(
            f"MRI slice ordering (freq, phase, slice index):\t{self.freq_dim} {self.phase_dim} {self.slice_dim}"
        )
        print(f"Start/end slice:\t\t\t\t{self.slice_start} {self.slice_end}")
        print(f"Slice duration:\t\t\t\t\t{self.slice_duration}")

        # Orientation
        print(f"Q factor:\t\t\t\t\t{self.qfac}")
        print(f"Qform transform code:\t\t\t\t{self.qform_code} ({self.qform_code.name})")
        quatern_str = " ".join(str(q) for q in self.quatern)
        print(f"Quaternion b,c,d params:\t\t\t{quatern_str}")
        qoffset_str = " ".join(str(q) for q in self.qoffset)
        print(f"Quaternion x,y,z shifts:\t\t\t{qoffset_str}")

        print(f"Affine transform code:\t\t\t\t{self.sform_code} ({self.sform_code.name})")
        srow_x_str = " ".join(str(s) for s in self.srow_x)
        print(f"1st row affine transform:\t\t\t{srow_x_str}")
        srow_y_str = " ".join(str(s) for s in self.srow_y)
        print(f"2nd row affine transform:\t\t\t{srow_y_str}")
        srow_z_str = " ".join(str(s) for s in self.srow_z)
        print(f"3rd row affine transform:\t\t\t{srow_z_str}")

        # Comments
        print(f"Description:\t\t\t\t\t{repr(self.descrip)}")
        print(f"Intent name:\t\t\t\t\t{repr(self.intent_name)}")
        print(f"Auxiliary file:\t\t\t\t\t{repr(self.aux_file)}")
        print(f"Extension byte 1:\t\t\t\t{self.extension[0]}")

        # Extensions
        if self.extension[0] != 0:
            print(f"\n\nExtensions")
            print(
                "----------------------------------------------------------------------"
            )
            print("#\tCode\tSize")
            for i, (size, code) in enumerate(self.extensions_list):
                print(f"{i+1}\t{code}\t{size}")

        # Unused stuff
        print("\n\nUnused Fields")
        print("----------------------------------------------------------------------")
        print(f"Data type string:\t\t\t{self.data_type_string}")
        print(f"db_name:\t\t\t\t{self.db_name}")
        print(f"extents:\t\t\t\t{self.extents}")
        print(f"session_error:\t\t\t\t{self.session_error}")
        print(f"regular:\t\t\t\t{self.regular}")
        print(f"glmax/glmin:\t\t\t\t{self.glmax} {self.glmin}")
        print(
            f"Extension bytes 2-4:\t\t\t{int(self.extension[1])} {int(self.extension[2])} {int(self.extension[3])}"
        )

    def read_double_vol(self, t: int = 0) -> np.ndarray:
        """
        Read one 3D volume from disk and return as numpy array.

        Args:
            t: Time dimension index (0-based)

        Returns:
            3D numpy array with shape (Z, Y, X)
        """
        # Handle 2D case
        ZZZ = self.ZDIM if self.dim[0] > 2 else 1

        # Read raw volume data
        blob = self.read_vol_blob(t)

        # Create reader for the blob
        reader = EndianCorrectReader(BytesIO(blob), self.big_endian)

        # Initialize output array
        data = np.zeros((ZZZ, self.YDIM, self.XDIM), dtype=np.float64)

        # Read data based on datatype
        if self.datatype in [
            NiftiDataType.NIFTI_TYPE_INT8,
            NiftiDataType.NIFTI_TYPE_UINT8,
        ]:
            for k in range(ZZZ):
                for j in range(self.YDIM):
                    for i in range(self.XDIM):
                        if self.datatype == NiftiDataType.NIFTI_TYPE_INT8:
                            val = reader.read_int8()
                        else:
                            val = reader.read_uint8()

                        data[k, j, i] = float(val)
                        if self.scl_slope != 0:
                            data[k, j, i] = (
                                data[k, j, i] * self.scl_slope + self.scl_inter
                            )

        elif self.datatype in [
            NiftiDataType.NIFTI_TYPE_INT16,
            NiftiDataType.NIFTI_TYPE_UINT16,
        ]:
            for k in range(ZZZ):
                for j in range(self.YDIM):
                    for i in range(self.XDIM):
                        if self.datatype == NiftiDataType.NIFTI_TYPE_INT16:
                            val = reader.read_int16()
                        else:
                            val = reader.read_uint16()

                        data[k, j, i] = float(val)
                        if self.scl_slope != 0:
                            data[k, j, i] = (
                                data[k, j, i] * self.scl_slope + self.scl_inter
                            )

        elif self.datatype in [
            NiftiDataType.NIFTI_TYPE_INT32,
            NiftiDataType.NIFTI_TYPE_UINT32,
        ]:
            for k in range(ZZZ):
                for j in range(self.YDIM):
                    for i in range(self.XDIM):
                        if self.datatype == NiftiDataType.NIFTI_TYPE_INT32:
                            val = reader.read_int32()
                        else:
                            val = reader.read_uint32()

                        data[k, j, i] = float(val)
                        if self.scl_slope != 0:
                            data[k, j, i] = (
                                data[k, j, i] * self.scl_slope + self.scl_inter
                            )

        elif self.datatype in [
            NiftiDataType.NIFTI_TYPE_INT64,
            NiftiDataType.NIFTI_TYPE_UINT64,
        ]:
            for k in range(ZZZ):
                for j in range(self.YDIM):
                    for i in range(self.XDIM):
                        if self.datatype == NiftiDataType.NIFTI_TYPE_INT64:
                            val = reader.read_int64()
                        else:
                            val = reader.read_uint64()

                        data[k, j, i] = float(val)
                        if self.scl_slope != 0:
                            data[k, j, i] = (
                                data[k, j, i] * self.scl_slope + self.scl_inter
                            )

        elif self.datatype == NiftiDataType.NIFTI_TYPE_FLOAT32:
            for k in range(ZZZ):
                for j in range(self.YDIM):
                    for i in range(self.XDIM):
                        val = reader.read_float32()
                        data[k, j, i] = val
                        if self.scl_slope != 0:
                            data[k, j, i] = (
                                data[k, j, i] * self.scl_slope + self.scl_inter
                            )

        elif self.datatype == NiftiDataType.NIFTI_TYPE_FLOAT64:
            for k in range(ZZZ):
                for j in range(self.YDIM):
                    for i in range(self.XDIM):
                        val = reader.read_float64()
                        data[k, j, i] = val
                        if self.scl_slope != 0:
                            data[k, j, i] = (
                                data[k, j, i] * self.scl_slope + self.scl_inter
                            )

        else:
            raise ValueError(
                f"Unsupported datatype: {self.datatype.name}"
            )

        reader.close()
        return data

    def read_vol_blob(self, t: int = 0) -> bytes:
        """
        Read one 3D volume blob from disk as raw bytes.

        Args:
            t: Time dimension index (0-based)

        Returns:
            Raw bytes of the volume data
        """
        # Handle 2D case
        ZZZ = self.ZDIM if self.dim[0] > 2 else 1

        blob_size = (
            self.XDIM * self.YDIM * ZZZ * NiftiDataType.bytes_per_voxel(self.datatype)
        )

        skip_head = int(self.vox_offset)
        skip_data = t * blob_size

        if self.ds_datname.endswith(".gz"):
            with gzip.open(self.ds_datname, "rb") as f:
                f.seek(skip_head + skip_data)
                return f.read(blob_size)
        else:
            with open(self.ds_datname, "rb") as f:
                f.seek(skip_head + skip_data)
                return f.read(blob_size)

    def read_double_timecourse(self, x: int, y: int, z: int) -> np.ndarray:
        """
        Read timecourse for a single voxel from 4D dataset.

        Args:
            x, y, z: Voxel coordinates (0-based)

        Returns:
            1D numpy array of timecourse values
        """
        ZZZ = self.ZDIM if self.dim[0] > 2 else 1

        data = np.zeros(self.TDIM, dtype=np.float64)

        skip_head = int(self.vox_offset)
        voxel_offset = (
            z * self.XDIM * self.YDIM + y * self.XDIM + x
        ) * NiftiDataType.bytes_per_voxel(self.datatype)
        vol_size = (
            self.XDIM * self.YDIM * ZZZ * NiftiDataType.bytes_per_voxel(self.datatype)
        )

        if self.ds_datname.endswith(".gz"):
            reader = EndianCorrectReader(self.ds_datname, self.big_endian)
        else:
            reader = EndianCorrectReader(self.ds_datname, self.big_endian)

        reader.skip(skip_head + voxel_offset)

        for t in range(self.TDIM):
            if t > 0:
                reader.skip(vol_size - NiftiDataType.bytes_per_voxel(self.datatype))

            # Read voxel value based on datatype
            if self.datatype == NiftiDataType.NIFTI_TYPE_INT8:
                val = float(reader.read_int8())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_UINT8:
                val = float(reader.read_uint8())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_INT16:
                val = float(reader.read_int16())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_UINT16:
                val = float(reader.read_uint16())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_INT32:
                val = float(reader.read_int32())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_UINT32:
                val = float(reader.read_uint32())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_INT64:
                val = float(reader.read_int64())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_UINT64:
                val = float(reader.read_uint64())
            elif self.datatype == NiftiDataType.NIFTI_TYPE_FLOAT32:
                val = reader.read_float32()
            elif self.datatype == NiftiDataType.NIFTI_TYPE_FLOAT64:
                val = reader.read_float64()
            else:
                raise ValueError(
                    f"Unsupported datatype: {self.datatype.name}"
                )

            # Apply scaling
            if self.scl_slope != 0:
                val = val * self.scl_slope + self.scl_inter

            data[t] = val

        reader.close()
        return data

    def set_dims(
        self,
        ndim: int,
        x: int,
        y: int,
        z: int,
        t: int = 1,
        d5: int = 1,
        d6: int = 1,
        d7: int = 1,
    ):
        """Set dataset dimensions"""
        self.dim[0] = ndim
        self.dim[1] = x
        self.dim[2] = y
        self.dim[3] = z
        self.dim[4] = t
        self.dim[5] = d5
        self.dim[6] = d6
        self.dim[7] = d7

        self.XDIM = x
        self.YDIM = y
        self.ZDIM = z
        self.TDIM = t
        self.DIM5 = d5
        self.DIM6 = d6
        self.DIM7 = d7

    def set_datatype(self, datatype: int):
        """Set dataset datatype and update bitpix accordingly"""
        self.datatype = datatype
        self.bitpix = NiftiDataType.bytes_per_voxel(datatype) * 8

    def get_datatype(self) -> int:
        """Get dataset datatype"""
        return self.datatype

    def get_bitpix(self) -> int:
        """Get bits per pixel"""
        return self.bitpix


# Example usage and test functions
def main():
    """Example usage of the Nifti1Dataset class"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nifti1dataset.py <nifti_file>")
        return

    # Create dataset object
    nds = Nifti1Dataset(sys.argv[1])

    try:
        # Read and print header
        nds.read_header()
        nds.print_header()

        # If it's a 4D dataset, read a sample timecourse
        if nds.TDIM > 1 and nds.exists():
            print(f"\nReading timecourse for voxel (0, 0, 0)...")
            timecourse = nds.read_double_timecourse(0, 0, 0)
            print(f"First 10 timepoints: {timecourse[:10]}")

        # If it's a 3D dataset, read first volume
        elif nds.ZDIM > 0 and nds.exists():
            print(f"\nReading first volume...")
            vol = nds.read_double_vol(0)
            print(f"Volume shape: {vol.shape}")
            print(
                f"Volume stats: min={vol.min():.3f}, max={vol.max():.3f}, mean={vol.mean():.3f}"
            )
            print(vol)

    except Exception as e:
        print(f"Error reading NIfTI file: {e}")


if __name__ == "__main__":
    main()
