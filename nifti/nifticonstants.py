from enum import IntEnum

class NiftiConstants:
    ANZ_HDR_EXT = ".hdr"
    ANZ_DAT_EXT = ".img"
    NII_EXT = ".nii"
    GZIP_EXT = ".gz"
    ANZ_HDR_SIZE = 348
    NII_HDR_SIZE = 352
    EXT_KEY_SIZE = 8
    NII_MAGIC_STRING = "n+1\0"
    ANZ_MAGIC_STRING = "ni1\0"


class NiftiDataType(IntEnum):
    DT_NONE = 0
    DT_BINARY = 1
    NIFTI_TYPE_UINT8 = 2
    NIFTI_TYPE_INT16 = 4
    NIFTI_TYPE_INT32 = 8
    NIFTI_TYPE_FLOAT32 = 16
    NIFTI_TYPE_COMPLEX64 = 32
    NIFTI_TYPE_FLOAT64 = 64
    NIFTI_TYPE_RGB24 = 128
    DT_ALL = 255
    NIFTI_TYPE_INT8 = 256
    NIFTI_TYPE_UINT16 = 512
    NIFTI_TYPE_UINT32 = 768
    NIFTI_TYPE_INT64 = 1024
    NIFTI_TYPE_UINT64 = 1280
    NIFTI_TYPE_FLOAT128 = 1536
    NIFTI_TYPE_COMPLEX128 = 1792
    NIFTI_TYPE_COMPLEX256 = 2048

    @staticmethod
    def bytes_per_voxel(datatype: int) -> int:
        mapping = {
            NiftiDataType.DT_NONE: 0,
            NiftiDataType.DT_BINARY: -1,  # 1 bit
            NiftiDataType.NIFTI_TYPE_UINT8: 1,
            NiftiDataType.NIFTI_TYPE_INT16: 2,
            NiftiDataType.NIFTI_TYPE_INT32: 4,
            NiftiDataType.NIFTI_TYPE_FLOAT32: 4,
            NiftiDataType.NIFTI_TYPE_COMPLEX64: 8,
            NiftiDataType.NIFTI_TYPE_FLOAT64: 8,
            NiftiDataType.NIFTI_TYPE_RGB24: 3,
            NiftiDataType.DT_ALL: 0,
            NiftiDataType.NIFTI_TYPE_INT8: 1,
            NiftiDataType.NIFTI_TYPE_UINT16: 2,
            NiftiDataType.NIFTI_TYPE_UINT32: 4,
            NiftiDataType.NIFTI_TYPE_INT64: 8,
            NiftiDataType.NIFTI_TYPE_UINT64: 8,
            NiftiDataType.NIFTI_TYPE_FLOAT128: 16,
            NiftiDataType.NIFTI_TYPE_COMPLEX128: 16,
            NiftiDataType.NIFTI_TYPE_COMPLEX256: 32,
        }
        # Convert int to enum if necessary
        if isinstance(datatype, int):
            try:
                datatype = NiftiDataType(datatype)
            except ValueError:
                return 0
        return mapping.get(datatype, 0)


class NiftiUnit(IntEnum):
    NIFTI_UNITS_UNKNOWN = 0
    NIFTI_UNITS_METER = 1
    NIFTI_UNITS_MM = 2
    NIFTI_UNITS_MICRON = 3
    NIFTI_UNITS_SEC = 8
    NIFTI_UNITS_MSEC = 16
    NIFTI_UNITS_USEC = 24
    NIFTI_UNITS_HZ = 32
    NIFTI_UNITS_PPM = 40


class NiftiIntent(IntEnum):
    NIFTI_INTENT_NONE = 0
    NIFTI_INTENT_CORREL = 2
    NIFTI_INTENT_TTEST = 3
    NIFTI_INTENT_FTEST = 4
    NIFTI_INTENT_ZSCORE = 5
    NIFTI_INTENT_CHISQ = 6
    NIFTI_INTENT_BETA = 7
    NIFTI_INTENT_BINOM = 8
    NIFTI_INTENT_GAMMA = 9
    NIFTI_INTENT_POISSON = 10
    NIFTI_INTENT_NORMAL = 11
    NIFTI_INTENT_FTEST_NONC = 12
    NIFTI_INTENT_CHISQ_NONC = 13
    NIFTI_INTENT_LOGISTIC = 14
    NIFTI_INTENT_LAPLACE = 15
    NIFTI_INTENT_UNIFORM = 16
    NIFTI_INTENT_TTEST_NONC = 17
    NIFTI_INTENT_WEIBULL = 18
    NIFTI_INTENT_CHI = 19
    NIFTI_INTENT_INVGAUSS = 20
    NIFTI_INTENT_EXTVAL = 21
    NIFTI_INTENT_PVAL = 22
    NIFTI_INTENT_ESTIMATE = 1001
    NIFTI_INTENT_LABEL = 1002
    NIFTI_INTENT_NEURONAME = 1003
    NIFTI_INTENT_GENMATRIX = 1004
    NIFTI_INTENT_SYMMATRIX = 1005
    NIFTI_INTENT_DISPVECT = 1006
    NIFTI_INTENT_VECTOR = 1007
    NIFTI_INTENT_POINTSET = 1008
    NIFTI_INTENT_TRIANGLE = 1009
    NIFTI_INTENT_QUATERNION = 1010


class NiftiSliceOrder(IntEnum):
    NIFTI_SLICE_UNKNOWN = 0
    NIFTI_SLICE_SEQ_INC = 1
    NIFTI_SLICE_SEQ_DEC = 2
    NIFTI_SLICE_ALT_INC = 3
    NIFTI_SLICE_ALT_DEC = 4


class NiftiXformCode(IntEnum):
    NIFTI_XFORM_UNKNOWN = 0
    NIFTI_XFORM_SCANNER_ANAT = 1
    NIFTI_XFORM_ALIGNED_ANAT = 2
    NIFTI_XFORM_TALAIRACH = 3
    NIFTI_XFORM_MNI_152 = 4
