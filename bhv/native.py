from bhv.abstract import AbstractBHV, DIMENSION
from bhv.cnative import CNativePackedBHV, _DIMENSION

assert DIMENSION == _DIMENSION()


class NativePackedBHV(AbstractBHV):
    def __init__(self, native_bhv):
        self.ins = native_bhv

    @classmethod
    def rand(cls):
        return NativePackedBHV(CNativePackedBHV.rand())

    @classmethod
    def random(cls, p):
        return NativePackedBHV(CNativePackedBHV.random(p))

    def __or__(self, other):
        return NativePackedBHV(self.ins | other.ins)

    def __and__(self, other):
        return NativePackedBHV(self.ins & other.ins)

    def __xor__(self, other):
        return NativePackedBHV(self.ins ^ other.ins)

    def __invert__(self):
        return NativePackedBHV(~self.ins)

    def select(self, when1, when0):
        return NativePackedBHV(self.ins.select(when1.ins, when0.ins))

    @classmethod
    def majority(cls, xs):
        return NativePackedBHV(CNativePackedBHV.majority([x.ins for x in xs]))

    def active(self):
        return self.ins.active()

    def hamming(self, other):
        return self.ins.hamming(other.ins)

