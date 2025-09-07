import gzip
import numpy as np
import nibabel as nib
from typing import Tuple


class Int32Slices:
    """Class to extract 2D slices from a 3D NIfTI volume as int32 arrays. 
    Based on Int32Slices.java from VisuAlign.
    """

    def __init__(self, nifti_file: str):
        self.nifti_file = nifti_file
        self.n1d = nib.load(nifti_file)

        hdr = self.n1d.header
        self.type = hdr.get_data_dtype().type
        self.XDIM, self.YDIM, self.ZDIM = self.n1d.shape[:3]

        # Bytes per voxel
        self.BPV = hdr["bitpix"].item() // 8
        if self.BPV > 4:
            raise ValueError(f"Unsupported voxel size: {self.BPV} bytes")

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
        origin: Tuple[int, int ,int],
        horizontal_axis: Tuple[int, int, int],
        vertical_axis: Tuple[int, int, int],
        grayscale=True
    ) -> np.ndarray:
        """
        Extract a 2D slice from the 3D atlas volume.
        - ox, oy, oz: slice origin
        - ux, uy, uz: horizontal axis vector
        - vx, vy, vz: vertical axis vector
        - grayscale: if True, normalize to 0-65535
        """

        ox, oy, oz = origin
        ux, uy, uz = horizontal_axis
        vx, vy, vz = vertical_axis

        scale = 2.5 if self.blob10 is not None else 1.0
        ox, oy, oz = ox * scale, oy * scale, oz * scale
        ux, uy, uz = ux * scale, uy * scale, uz * scale
        vx, vy, vz = vx * scale, vy * scale, vz * scale

        width = int(np.sqrt(ux * ux + uy * uy + uz * uz)) + 1
        height = int(np.sqrt(vx * vx + vy * vy + vz * vz)) + 1
        slice_arr = np.zeros((height, width), dtype=np.int32)

        for y in range(height):
            hx = ox + vx * y / height
            hy = oy + vy * y / height
            hz = oz + vz * y / height
            for x in range(width):
                lx = int(hx + ux * x / width)
                ly = int(hy + uy * x / width)
                lz = int(hz + uz * x / width)

                if (0 <= lx < self.XDIM and
                    0 <= ly < self.YDIM and
                    0 <= lz < self.ZDIM):

                    if self.blob10 is not None:
                        slice_arr[y, x] = self.blob10[lz, self.YDIM - 1 - ly, lx]
                    else:
                        slice_arr[y, x] = int(self.data[lx, ly, lz])

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

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python int32slices.py <nifti_file>")
        return

    nifti_file = sys.argv[1]
    atlas = Int32Slices(nifti_file)

    # Example slice parameters (origin at center, along x/y axes)
    ox, oy, oz = atlas.XDIM // 2, atlas.YDIM // 2, atlas.ZDIM // 2
    ux, uy, uz = 100, 0, 0
    vx, vy, vz = 0, 100, 0

    slice_arr = atlas.get_int32_slice(ox, oy, oz, ux, uy, uz, vx, vy, vz)

    print(f"Extracted slice shape: {slice_arr.shape}")
    print(slice_arr)

if __name__ == "__main__":
    main()