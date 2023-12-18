from enum import Enum, unique

@unique
class QuantMode(Enum):
    FullPrecision = 0
    Quantization = 1