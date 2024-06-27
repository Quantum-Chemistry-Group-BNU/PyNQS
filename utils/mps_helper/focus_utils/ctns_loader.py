import numpy as np
import struct

class tensor3():
    def __init__(self):
        self.rows = None
        self.cols = None
        self.mids = None
        self.qrow = None
        self.qcol = None
        self.qmid = None
        self.nblks = None
        self.offset = None
        self.size = None
        self.data = None

    def load(self,fname,data,off,dtype):
        # rows
        self.rows = struct.unpack('i', data[off:off+4])[0]
        off += 4
        self.qrow = np.fromfile(fname, dtype=np.int32, offset=off, count=3*self.rows)
        self.qrow = self.qrow.reshape(self.rows,3)
        off += self.qrow.nbytes
        # cols
        self.cols = struct.unpack('i', data[off:off+4])[0]
        off += 4
        self.qcol = np.fromfile(fname, dtype=np.int32, offset=off, count=3*self.cols)
        self.qcol = self.qcol.reshape(self.cols,3)
        off += self.qcol.nbytes
        # mids
        self.mids = struct.unpack('i', data[off:off+4])[0]
        off += 4
        self.qmid = np.fromfile(fname, dtype=np.int32, offset=off, count=3*self.mids)
        self.qmid = self.qmid.reshape(self.mids,3)
        off += self.qmid.nbytes
        # offset
        self.nblks = self.rows*self.cols*self.mids
        self.offset = np.fromfile(fname, dtype=np.uint64, offset=off, count=self.nblks)
        self.offset = self.offset.reshape(self.rows,self.cols,self.mids)
        off += self.offset.nbytes
        # data
        self.size = struct.unpack('N', data[off:off+8])[0]
        off += 8
        self.data = np.fromfile(fname, dtype=dtype, offset=off, count=self.size)
        off += self.data.nbytes
        return off 

    def prt(self, name=''):
        print('### site '+name+' ###')
        print('qrow=',self.qrow)
        print('qcol=',self.qcol)
        print('qmid=',self.qmid)
        print('offset=',self.offset)
        print('size=',self.size)
        print('data',self.data)
        print()
        return 0

    def todense(self):
        drow = self.qrow[:,2]
        dcol = self.qcol[:,2]
        dmid = self.qmid[:,2]
        dim_row = np.sum(drow)
        dim_col = np.sum(dcol)
        dim_mid = np.sum(dmid)
        dtensor = np.zeros((dim_mid,dim_col,dim_row), dtype=self.data.dtype) # nrl in C order
        off_row = [0] + list(np.cumsum(drow))
        off_col = [0] + list(np.cumsum(dcol))
        off_mid = [0] + list(np.cumsum(dmid))
        for r in range(self.rows):
            dr = self.qrow[r,2] 
            for c in range(self.cols):
                dc = self.qcol[c,2]
                for m in range(self.mids):
                    dm = self.qmid[m,2]
                    off = self.offset[r,c,m] # lrn in C order 
                    if(off == 0): continue
                    blksize = np.uint64(dr*dc*dm)
                    off -= np.uint64(1) 
                    sta_r = off_row[r]
                    sta_c = off_col[c]
                    sta_m = off_mid[m]
                    # data stored lrn in F order 
                    dtensor[sta_m:sta_m+dm,sta_c:sta_c+dc,sta_r:sta_r+dr] = \
                            self.data[off:off+blksize].reshape(dm,dc,dr).copy()
        dtensor = dtensor.transpose(2,0,1).copy() # nrl->lnr convention
        return dtensor


class ctns_info():
    def __init__(self):
        self.ntotal = None
        self.rsites = None

    def load(self,fname, dtype=np.float64):
        data = open(fname, 'rb').read()
        off = 4
        self.ntotal = struct.unpack('i', data[:off])[0]
        self.rsites = [None]*self.ntotal
        for i in range(self.ntotal):
            self.rsites[i] = tensor3();
            off = self.rsites[i].load(fname,data,off,dtype)
        return 0

    def prt(self):
        print('ntotal=',self.ntotal)
        for i in range(self.ntotal):
            self.rsites[i].prt(str(i))
        return 0 

    # lnr convention
    def toMPSdense(self):
        sites = [None]*self.ntotal
        for i in range(self.ntotal):
            sites[self.ntotal-1-i] = self.rsites[i].todense()
        return sites

if __name__ == '__main__':

    ctns = ctns_info()
    ctns.load('rcanon_isweep3.info.bin')
    ctns.prt()

    for i in range(ctns.ntotal):
        print(ctns.rsites[i].todense())
