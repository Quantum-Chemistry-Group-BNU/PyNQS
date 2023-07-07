import numpy as np

from numpy import ndarray
from typing import List, Tuple, NewType, overload

# from libs.qubic.qtensor import Qbond, Qinfo2, Qsym, Stensor2

qsym = NewType("qsym", Tuple[int, int, int])
dim = NewType("dim", int)

class Qsym:
    """
    Qsym: Tuple[isym, ne, tm]
    """
    @overload
    def __init__(self)->None: ...

    @overload
    def __init__(self, isym: int)->None: ...

    @overload
    def __init__(self, isym: int, ne: int)->None: ...

    @overload
    def __init__(self, isym:int, ne: int, tm: int)->None: ...

    def is_zero(self)->bool: """ ne == 0 && tm == 0 """

    def is_nonzero(self)->bool: """ ne != 0 || tm != 0 """

    def isym(self)->int: """ isym """

    def ne(self)->int: """ ne = na + nb """

    def tm(self)->int: """tm = na - nb"""

    def data(self)->qsym: """tuple(isym, ne, tm)"""

class Qbond:
    """
    Qbond: List[Tuple[qsym, int]]
    """
    def __init__(self) -> None: ...

    def print(self)->None: """print Qbond class information"""

    def data(self)->Tuple[List[qsym], List[dim]]: """Qbond raw data"""

    def size(self)->int: """the length of Qbond2"""

    def get_sym(self, n: int)->int: """n-th Qsym"""

    def get_dim(self, n: int)->int: """n-th dim"""

    def __getitem__(self, n: int)->Tuple[qsym, dim]:"""return n-th Qsym and dim"""

    def __len__(self)->int: """the length of Qbond2"""

class Qinfo2:
    """
    2D sparse tensor/matrix structural information
    """

    def __init__(self) -> None: ...

    def print(self)->None: """print Qinfo2 class information"""

    @property
    def sym(self)->Qsym:...
    
    @sym.getter
    def sym(self)->Qsym:...

    @property
    def qrow(self)->Qbond:...

    @qrow.getter
    def qrow(self)->Qbond:...

    @property
    def qcol(self)->Qbond:...

    @qcol.getter
    def qcol(self)->Qbond:...

    @property
    def nnzaddr(self)->ndarray[np.int64]:"the address of nonzero block, C oder"

    @nnzaddr.getter
    def nnzaddr(self)->ndarray[np.int64]:...

    @property
    def size(self)->int:"""the number of nonzero data size """

    @size.getter
    def size(self)->int:...



class Stensor2:
    """
    2D sparse tensor/matrix
    """
    def __init__(self)->None: ...

    def rows(self)->int: ...

    def cols(self)->int: ...

    def data(self)->ndarray[np.double]: """ 2D Sparse tensor nonzero data stored by a 1D array """
    
    def size(self)->int: """the number of nonzero data size """

    def info(self)->Qinfo2: """block structural information """

    def shape(self)->Tuple[int, int]:"""sparse matrix block shape"""


