from typing import NewType, Tuple, Dict

NSz = NewType("NSz", Tuple[int, int])
NSz_index = NewType("NSz_dict", Dict[NSz, int])