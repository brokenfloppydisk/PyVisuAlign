import struct, gzip

class EndianCorrectReader:
    """Handles endian-correct reading of binary data
    
    Based on EndianCorrectInputStream: https://github.com/NIFTI-Imaging/nifti_java/blob/master/niftijlib/EndianCorrectInputStream.java
    """

    def __init__(self, file_or_stream, big_endian: bool = True):
        if isinstance(file_or_stream, str):
            if file_or_stream.endswith(".gz"):
                self.stream = gzip.open(file_or_stream, "rb")
            else:
                self.stream = open(file_or_stream, "rb")
        else:
            self.stream = file_or_stream

        self.big_endian = big_endian
        self.endian_char = ">" if big_endian else "<"

    def close(self):
        """Close the stream"""
        if hasattr(self.stream, "close"):
            self.stream.close()

    def read(self, size: int) -> bytes:
        """Read bytes from stream"""
        return self.stream.read(size)

    def skip(self, size: int):
        """Skip bytes in stream"""
        self.stream.seek(size, 1)

    def read_int8(self) -> int:
        """Read signed 8-bit integer"""
        return struct.unpack("b", self.stream.read(1))[0]

    def read_uint8(self) -> int:
        """Read unsigned 8-bit integer"""
        return struct.unpack("B", self.stream.read(1))[0]

    def read_int16(self) -> int:
        """Read signed 16-bit integer with correct endianness"""
        return struct.unpack(f"{self.endian_char}h", self.stream.read(2))[0]

    def read_uint16(self) -> int:
        """Read unsigned 16-bit integer with correct endianness"""
        return struct.unpack(f"{self.endian_char}H", self.stream.read(2))[0]

    def read_int32(self) -> int:
        """Read signed 32-bit integer with correct endianness"""
        return struct.unpack(f"{self.endian_char}i", self.stream.read(4))[0]

    def read_uint32(self) -> int:
        """Read unsigned 32-bit integer with correct endianness"""
        return struct.unpack(f"{self.endian_char}I", self.stream.read(4))[0]

    def read_int64(self) -> int:
        """Read signed 64-bit integer with correct endianness"""
        return struct.unpack(f"{self.endian_char}q", self.stream.read(8))[0]

    def read_uint64(self) -> int:
        """Read unsigned 64-bit integer with correct endianness"""
        return struct.unpack(f"{self.endian_char}Q", self.stream.read(8))[0]

    def read_float32(self) -> float:
        """Read 32-bit float with correct endianness"""
        return struct.unpack(f"{self.endian_char}f", self.stream.read(4))[0]

    def read_float64(self) -> float:
        """Read 64-bit float with correct endianness"""
        return struct.unpack(f"{self.endian_char}d", self.stream.read(8))[0]

    def read_string(self, size: int) -> str:
        """Read null-terminated string"""
        data = self.stream.read(size)
        # Remove null bytes and decode
        return data.rstrip(b"\x00").decode("ascii", errors="ignore")


class EndianCorrectWriter:
    """Handles endian-correct writing of binary data.
    
    Based on EndianCorrectOutputStream: https://github.com/NIFTI-Imaging/nifti_java/blob/master/niftijlib/EndianCorrectOutputStream.java
    """

    def __init__(self, file_or_stream, big_endian: bool = True):
        if isinstance(file_or_stream, str):
            self.stream = open(file_or_stream, "wb")
        else:
            self.stream = file_or_stream

        self.big_endian = big_endian
        self.endian_char = ">" if big_endian else "<"

    def close(self):
        """Close the stream"""
        if hasattr(self.stream, "close"):
            self.stream.close()

    def write_int8(self, value: int):
        """Write signed 8-bit integer"""
        self.stream.write(struct.pack("b", value))

    def write_uint8(self, value: int):
        """Write unsigned 8-bit integer"""
        self.stream.write(struct.pack("B", value))

    def write_int16(self, value: int):
        """Write signed 16-bit integer with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}h", value))

    def write_uint16(self, value: int):
        """Write unsigned 16-bit integer with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}H", value))

    def write_int32(self, value: int):
        """Write signed 32-bit integer with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}i", value))

    def write_uint32(self, value: int):
        """Write unsigned 32-bit integer with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}I", value))

    def write_int64(self, value: int):
        """Write signed 64-bit integer with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}q", value))

    def write_uint64(self, value: int):
        """Write unsigned 64-bit integer with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}Q", value))

    def write_float32(self, value: float):
        """Write 32-bit float with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}f", value))

    def write_float64(self, value: float):
        """Write 64-bit float with correct endianness"""
        self.stream.write(struct.pack(f"{self.endian_char}d", value))

    def write_string(self, value: str, size: int):
        """Write string padded to specific size"""
        data = value.encode("ascii")[:size]
        # Pad with zeros
        padded = data + b"\x00" * (size - len(data))
        self.stream.write(padded)
