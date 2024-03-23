from enum import Enum, unique

@unique
class QuantMode(Enum):
    FullPrecision = 0                       # 全精度
    SymmQuantization = 1                    # 对称量化
    AsymmQuantization = 2                   # 非对称量化
    AsymmQuantizationWithCompensation = 3   # 非对称量化 + 内置补偿