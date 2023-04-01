from .abstract import *
from typing import Tuple
import numpy as np


class NumPyBoolBHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.random.randint(0, high=2, size=DIMENSION, dtype=np.bool_))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.random.binomial(1, active, DIMENSION))

    def select(self, when1: 'NumPyBoolBHV', when0: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.where(self.data, when1.data, when0.data))

    @classmethod
    def majority(cls, vs: list['NumPyBoolBHV']) -> 'NumPyBoolBHV':
        data = [v.data for v in vs]
        extra = [cls.rand().data] if len(vs) % 2 == 0 else []

        tensor = np.stack(data + extra)
        counts = tensor.sum(axis=-2, dtype=np.uint8)

        threshold = (len(vs) + len(extra))//2

        return NumPyBoolBHV(np.greater(counts, threshold))

    def __eq__(self, other: 'NumPyBoolBHV') -> bool:
        return np.array_equal(self.data, other.data)

    def __xor__(self, other: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_not(self.data))

    def active(self) -> int:
        return int(np.sum(self.data))

    def pack8(self) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.packbits(self.data))

    def pack64(self) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.packbits(self.data).view(dtype=np.uint64))

NumPyBoolBHV.ZERO = NumPyBoolBHV(np.zeros(DIMENSION, dtype=np.bool_))
NumPyBoolBHV.ONE = NumPyBoolBHV(np.ones(DIMENSION, dtype=np.bool_))


class NumPyPacked8BHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.random.randint(0, 255, DIMENSION//8, dtype=np.uint8))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyPacked8BHV':
        return NumPyBoolBHV.random(active).pack8()

    def __eq__(self, other: 'NumPyPacked8BHV') -> bool:
        return np.array_equal(self.data, other.data)

    def __xor__(self, other: 'NumPyPacked8BHV') -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyPacked8BHV') -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyPacked8BHV') -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_not(self.data))

    def active(self) -> int:
        return sum(x.bit_count() for x in self.data)

    def unpack(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.unpackbits(self.data))

NumPyPacked8BHV.ZERO = NumPyPacked8BHV(np.zeros(DIMENSION//8, dtype=np.uint8))
NumPyPacked8BHV.ONE = NumPyPacked8BHV(np.full(DIMENSION//8, fill_value=255, dtype=np.uint8))


RAND = np.random.SFC64()
class NumPyPacked64BHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(RAND.random_raw(DIMENSION//64))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyPacked64BHV':
        return NumPyBoolBHV.random(active).pack64()

    @classmethod
    def _majority_via_unpacked(cls, vs: list['NumPyPacked64BHV']) -> 'NumPyPacked64BHV':
        return NumPyBoolBHV.majority([v.unpack() for v in vs]).pack64()

    @classmethod
    def _majority3(cls, a: 'NumPyPacked64BHV', b: 'NumPyPacked64BHV', c: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        # a:  1 0 0 1 0 1 1
        # b:  1 1 0 1 0 1 0
        # c:  1 0 1 0 0 0 0
        # M:  1 0 0 1 0 1 0

        # at least 2/3 agreeing on TRUE
        abh = a & b
        bch = b & c
        cah = c & a
        h = abh | bch | cah
        return h

    @classmethod
    def _majority3_via_ite(cls, a: 'NumPyPacked64BHV', b: 'NumPyPacked64BHV', c: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        # C:  1 0 0 1 0 1 1

        # |:  1 1 1 1 0 1 0
        # &:  1 0 0 0 0 0 0

        # M:  1 0 0 1 0 1 0

        return a.select(b | c, b & c)

    @classmethod
    def _majority5_via_ite(cls, a: 'NumPyPacked64BHV', b: 'NumPyPacked64BHV', c: 'NumPyPacked64BHV', d: 'NumPyPacked64BHV', e: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        mcde = cls._majority3_via_ite(c, d, e)
        return a.select(b.select(c | d | e,
                                 mcde),
                        b.select(mcde,
                                 c & d & e))

    @classmethod
    def _majority7_via_ite(cls, a: 'NumPyPacked64BHV', b: 'NumPyPacked64BHV', c: 'NumPyPacked64BHV', d: 'NumPyPacked64BHV', e: 'NumPyPacked64BHV', f: 'NumPyPacked64BHV', g: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        mefg = cls._majority3_via_ite(e, f, g)
        mcdefg = cls._majority5_via_ite(c, d, e, f, g)
        # the | and & can be optimized to a single application, too
        return a.select(b.select(c.select(d | e | f | g,  d.select(e | f | g, mefg)),  # d select with abc'
                                 mcdefg),
                        b.select(mcdefg,
                                 c.select(d.select(mefg, e & f & g), d & e & f & g)))  # d select with a'b'c

    @classmethod
    def _majority_via_ite(cls, vs: list['NumPyPacked64BHV']) -> 'NumPyPacked64BHV':
        def rec(sl: list['NumPyPacked64BHV']) -> Tuple['NumPyPacked64BHV', 'NumPyPacked64BHV', 'NumPyPacked64BHV']:
            if len(sl) == 3:
                (a, b, c) = sl
                top = b | c
                bottom = b & c
                maj = a.select(top, bottom)
                return maj, a | top, a & bottom
            else:
                (a, b, *rs) = sl
                pmaj, ptop, pbot = rec(rs)
                top_maj = b.select(ptop, pmaj)
                bot_maj = b.select(pmaj, pbot)
                maj = a.select(top_maj, bot_maj)
                top = a.select(b | ptop, top_maj)
                bot = a.select(bot_maj, b & pbot)
                return maj, top, bot
        return rec(vs)[0]

    @classmethod
    def _majority5(cls, a: 'NumPyPacked64BHV', b: 'NumPyPacked64BHV', c: 'NumPyPacked64BHV', d: 'NumPyPacked64BHV', e: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        # at least 3/5 agreeing on TRUE

        # 2*10 AND
        # 9*1 OR
        # (b ∧ d ∧ e) ∨ (a ∧ d ∧ e) ∨ (b ∧ c ∧ e) ∨ (a ∧ c ∧ e) ∨ (b ∧ c ∧ d) ∨ (a ∧ c ∧ d) ∨ (c ∧ d ∧ e) ∨ (a ∧ b ∧ e) ∨ (a ∧ b ∧ d) ∨ (a ∧ b ∧ c)

        ab = a & b
        cd = c & d
        de = d & e
        ce = c & e

        bde = b & de
        ade = a & de

        bce = b & ce
        ace = a & ce

        bcd = b & cd
        acd = a & cd
        cde = e & cd

        abe = ab & e
        abd = ab & d
        abc = ab & c

        h = bde | ade | bce | ace | bcd | acd | cde | abe | abd | abc
        return h

    @classmethod
    def _majority5_via_3(cls, a: 'NumPyPacked64BHV', b: 'NumPyPacked64BHV', c: 'NumPyPacked64BHV', d: 'NumPyPacked64BHV', e: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return cls._majority3(a, cls._majority3(b, c, d), cls._majority3(e, d, cls._majority3(c, b, a)))

    @classmethod
    def _majority7_via_3(cls, a: 'NumPyPacked64BHV', b: 'NumPyPacked64BHV', c: 'NumPyPacked64BHV', d: 'NumPyPacked64BHV', e: 'NumPyPacked64BHV', f: 'NumPyPacked64BHV', g: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        mdef = cls._majority3(d, e, f)
        mabc = cls._majority3(a, b, c)
        return cls._majority3(g, cls._majority3(c, mdef, cls._majority3(a, b, mdef)), cls._majority3(f, mabc, cls._majority3(d, e, mabc)))

    @classmethod
    def majority(cls, vs: list['NumPyPacked64BHV']) -> 'NumPyPacked64BHV':
        # potential breakup
        #
        # majority_functions = [majority_3, majority_5, ...]
        # def rec(vs):
        #     match len(vs) % ...:
        #         case ...: majority_functions[...](*[...])
        #

        cls._majority_via_unpacked(vs)

    def __eq__(self, other: 'NumPyPacked64BHV') -> bool:
        return np.array_equal(self.data, other.data)

    def __xor__(self, other: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_not(self.data))

    def active(self) -> int:
        return sum(x.bit_count() for x in self.data)

    def unpack(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.unpackbits(self.data.view(np.uint8)))

NumPyPacked64BHV.ZERO = NumPyPacked64BHV(np.zeros(DIMENSION//64, dtype=np.uint64))
NumPyPacked64BHV.ONE = NumPyPacked64BHV(np.full(DIMENSION//64, fill_value=-1, dtype=np.uint64))
