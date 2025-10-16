import gzip
import numpy as np
import numpy.typing as npt
import nibabel as nib
from typing import Tuple, Optional


class Int32Slices:
    """Class to extract 2D slices from a 3D NIfTI volume as int32 arrays. 
    Based on Int32Slices.java from VisuAlign.
    """

    def __init__(self, nifti_file: str):
        self.nifti_file = nifti_file
        image = nib.loadsave.load(nifti_file)

        if not isinstance(image, nib.nifti1.Nifti1Image):
            raise ValueError(f"Unsupported atlas type: {type(image)}")
        self.n1d: nib.nifti1.Nifti1Image = image

        hdr = self.n1d.header
        self.type = hdr.get_data_dtype().type
        self.XDIM, self.YDIM, self.ZDIM = self.n1d.shape[:3]

        # Special case: high-res cutlas "_10um.cutlas/labels.nii.gz"
        if nifti_file.endswith("_10um.cutlas/labels.nii.gz"):
            if hdr.endianness != "<" or self.type != np.uint32:
                raise ValueError("High-res atlas must be little-endian UINT32")

            # Preload entire dataset into blob10
            self.blob10 = np.zeros((self.ZDIM, self.YDIM, self.XDIM), dtype=np.int32)
            with gzip.open(nifti_file, "rb") as f:
                # Skip NIfTI header (352 bytes)
                f.read(352)
                for z in range(self.ZDIM):
                    for y in range(self.YDIM):
                        row = f.read(self.XDIM * 4)
                        self.blob10[z, y, :] = np.frombuffer(row, dtype="<i4")
        else:
            # Load via nibabel as memory map
            self.blob10 = None
            self.data = self.n1d.get_fdata(caching="unchanged")

    def get_int32_slice(
        self,
        origin: npt.ArrayLike,
        horizontal_axis: npt.ArrayLike,
        vertical_axis: npt.ArrayLike,
        grayscale: bool = True
    ) -> npt.NDArray[np.int32]:
        """
        Extract a 2D slice from the 3D atlas volume.
        - origin: 3-vector (ox, oy, oz) in voxel coordinates
        - horizontal_axis: 3-vector (ux, uy, uz) extent along slice X
        - vertical_axis: 3-vector (vx, vy, vz) extent along slice Y
        - grayscale: if True, normalize to 0-65535
        """
        o = np.asarray(origin, dtype=np.float64)
        u = np.asarray(horizontal_axis, dtype=np.float64)
        v = np.asarray(vertical_axis, dtype=np.float64)

        # Special case scaling for _10um cutlas
        if self.blob10 is not None:
            scale = 2.5
            o = o * scale
            u = u * scale
            v = v * scale
        
        width = max(1, int(np.ceil(np.linalg.norm(u))))
        height = max(1, int(np.ceil(np.linalg.norm(v))))

        slice_arr = np.zeros((height, width), dtype=np.int32)

        # Normalized coordinates in [0,1] across the extents (handle 1-pixel degenerate cases)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        scaled_y_coords = y_coords / height
        scaled_x_coords = x_coords / width

        lx = (o[0] + (v[0] * scaled_y_coords) + (u[0] * scaled_x_coords)).astype(np.int32)
        ly = (o[1] + (v[1] * scaled_y_coords) + (u[1] * scaled_x_coords)).astype(np.int32)
        lz = (o[2] + (v[2] * scaled_y_coords) + (u[2] * scaled_x_coords)).astype(np.int32)

        # In-bounds mask
        mask = (
            (lx >= 0) & (lx < self.XDIM) &
            (ly >= 0) & (ly < self.YDIM) &
            (lz >= 0) & (lz < self.ZDIM)
        )

        valid_lx = lx[mask]
        valid_ly = ly[mask]
        valid_lz = lz[mask]

        if self.blob10 is not None:
            slice_arr[mask] = self.blob10[valid_lz, self.YDIM - 1 - valid_ly, valid_lx]
        else:
            slice_arr[mask] = self.data[valid_lx, valid_ly, valid_lz].astype(np.int32)

        # Normalize if float or grayscale
        if np.issubdtype(self.type, np.floating):
            vals = slice_arr.astype(np.float32)
            minv, maxv = vals.min(), vals.max()
            slice_arr = ((vals - minv) / (maxv - minv) * 65535).astype(np.int32)

        elif grayscale and not np.issubdtype(self.type, np.dtype([("r", np.uint8), ("g", np.uint8), ("b", np.uint8)])):
            minv, maxv = slice_arr.min(), slice_arr.max()
            if maxv > minv:
                slice_arr = ((slice_arr - minv) / (maxv - minv) * 65535).astype(np.int32)

        return slice_arr

def main() -> None:
    import sys
    if len(sys.argv) != 2:
        print("Usage: python int32slices.py <nifti_file>")
        return

    nifti_file = sys.argv[1]
    atlas = Int32Slices(nifti_file)

    # Example slice parameters (origin at center, along x/y axes)
    origin = (atlas.XDIM // 2, atlas.YDIM // 2, atlas.ZDIM // 2)
    u = (100, 0, 0)
    v = (0, 100, 0)

    slice_arr = atlas.get_int32_slice(origin, u, v)

    print(f"Extracted slice shape: {slice_arr.shape}")
    print(slice_arr)

if __name__ == "__main__":
    main()
