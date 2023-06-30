#ifndef BHV_PACKED_H
#define BHV_PACKED_H

#include <random>
#include <cstring>
#include <algorithm>
#include "shared.h"
//#include "TurboSHAKE.h"
#include "TurboSHAKEopt/TurboSHAKE.h"

#include <immintrin.h>

namespace bhv {
    constexpr word_t ONE_WORD = std::numeric_limits<word_t>::max();
    constexpr bit_word_iter_t HALF_BITS_PER_WORD = BITS_PER_WORD/2;
    constexpr word_t HALF_WORD = ONE_WORD << HALF_BITS_PER_WORD;
    constexpr word_t OTHER_HALF_WORD = ~HALF_WORD;

    std::mt19937_64 rng;

    word_t * empty() {
        return (word_t *) malloc(BYTES);
    }

    word_t * zero() {
        return (word_t *) calloc(WORDS, sizeof(word_t));
    }

    word_t * one() {
        word_t * x = empty();
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = ONE_WORD;
        }
        return x;
    }

    word_t * half() {
        word_t * x = empty();
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = HALF_WORD;
        }
        return x;
    }

    void swap_halves_into(word_t * x, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ((x[i] & HALF_WORD) >> HALF_BITS_PER_WORD) | ((x[i] & OTHER_HALF_WORD) << HALF_BITS_PER_WORD);
        }
    }

    void rand_into(word_t * x) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = rng();
        }
    }

    void random_into(word_t * x, float_t p) {
        std::uniform_real_distribution<float> gen(0.0, 1.0);

        for (word_iter_t i = 0; i < WORDS; ++i) {
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (gen(rng) < p)
                    word |= 1UL << bit_id;
            }
            x[i] = word;
        }
    }

    word_t * rand() {
        word_t * x = empty();
        rand_into(x);
        return x;
    }

    word_t * random(float_t p) {
        word_t * x = empty();
        random_into(x, p);
        return x;
    }

    bit_iter_t active(word_t * x) {
        bit_iter_t total = 0;
        for (word_iter_t i = 0; i < WORDS; ++i) {
            total += __builtin_popcountl(x[i]);
        }
        return total;
    }

    bit_iter_t hamming(word_t * x, word_t * y) {
        bit_iter_t total = 0;
        for (word_iter_t i = 0; i < WORDS; ++i) {
            total += __builtin_popcountl(x[i] ^ y[i]);
        }
        return total;
    }

    bool eq(word_t * x, word_t * y) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            if (x[i] != y[i])
                return false;
        }
        return true;
    }


    void xor_into(word_t * x, word_t * y, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] ^ y[i];
        }
    }

    void and_into(word_t * x, word_t * y, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] & y[i];
        }
    }

    void or_into(word_t * x, word_t * y, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] | y[i];
        }
    }

    void invert_into(word_t * x, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ~x[i];
        }
    }

    template <typename N>
    N* generic_counts(word_t ** xs, N size) {
        N* totals = (N *) calloc(BITS, sizeof(N));

        for (N i = 0; i < size; ++i) {
            word_t * x = xs[i];

            for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
                bit_iter_t offset = word_id * BITS_PER_WORD;
                word_t word = x[word_id];
                for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                    totals[offset + bit_id] += ((word >> bit_id) & 1);
                }
            }
        }

        return totals;
    }

    template <typename N>
    word_t* generic_gt(N * totals, N threshold) {
        word_t * x = empty();

        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (threshold < totals[offset + bit_id])
                    word |= 1UL << bit_id;
            }
            x[word_id] = word;
        }
        free(totals);
        return x;
    }

    template<typename N> void threshold_into_counting_generic(word_t ** xs, size_t size, N threshold, word_t *dst) {

        N totals[BITS];
        memset(totals, 0, BITS*sizeof(N));

        for (N i = 0; i < size; ++i) {
            word_t * x = xs[i];

            for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
                bit_iter_t offset = word_id * BITS_PER_WORD;
                word_t word = x[word_id];
                for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                    totals[offset + bit_id] += ((word >> bit_id) & 1);
                }
            }
        }

        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (threshold < totals[offset + bit_id])
                    word |= 1UL << bit_id;
            }
            dst[word_id] = word;
        }
    }

    void threshold_into_counting_int8(word_t ** xs, size_t size, uint8_t threshold, word_t *dst) {

        uint8_t totals[BITS];
        memset(totals, 0, BITS*sizeof(uint8_t));

        for (uint8_t i = 0; i < size; ++i) {
            word_t * x = xs[i];
            word_t* totals_ptr = (word_t*)totals;

            for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
                word_t word = x[word_id];

                //Here we process 8 bits at a time, and there are 8 chunks of 8 bits in a 64 bit word
                for (int chunk = 0; chunk < 8; chunk++) {
                    uint8_t current_bits = word & 0xFF;
                    word = word >> 8;
                    word_t increment;

                    //NOTE: Inline case-statement performs quite a bit better than a static const table.
                    // not sure why, probably something to do with how the table gets packed in memory

                    //GOAT, this table *might* be wrong because of endian (byte order) issues.
                    // I'm just measuring performance right now.  We need some correctness tests.
                    switch (current_bits) {
                        case 0: increment = 0; break;
                        case 1: increment = 1; break;
                        case 2: increment = 256; break;
                        case 3: increment = 257; break;
                        case 4: increment = 65536; break;
                        case 5: increment = 65537; break;
                        case 6: increment = 65792; break;
                        case 7: increment = 65793; break;
                        case 8: increment = 16777216; break;
                        case 9: increment = 16777217; break;
                        case 10: increment = 16777472; break;
                        case 11: increment = 16777473; break;
                        case 12: increment = 16842752; break;
                        case 13: increment = 16842753; break;
                        case 14: increment = 16843008; break;
                        case 15: increment = 16843009; break;
                        case 16: increment = 4294967296; break;
                        case 17: increment = 4294967297; break;
                        case 18: increment = 4294967552; break;
                        case 19: increment = 4294967553; break;
                        case 20: increment = 4295032832; break;
                        case 21: increment = 4295032833; break;
                        case 22: increment = 4295033088; break;
                        case 23: increment = 4295033089; break;
                        case 24: increment = 4311744512; break;
                        case 25: increment = 4311744513; break;
                        case 26: increment = 4311744768; break;
                        case 27: increment = 4311744769; break;
                        case 28: increment = 4311810048; break;
                        case 29: increment = 4311810049; break;
                        case 30: increment = 4311810304; break;
                        case 31: increment = 4311810305; break;
                        case 32: increment = 1099511627776; break;
                        case 33: increment = 1099511627777; break;
                        case 34: increment = 1099511628032; break;
                        case 35: increment = 1099511628033; break;
                        case 36: increment = 1099511693312; break;
                        case 37: increment = 1099511693313; break;
                        case 38: increment = 1099511693568; break;
                        case 39: increment = 1099511693569; break;
                        case 40: increment = 1099528404992; break;
                        case 41: increment = 1099528404993; break;
                        case 42: increment = 1099528405248; break;
                        case 43: increment = 1099528405249; break;
                        case 44: increment = 1099528470528; break;
                        case 45: increment = 1099528470529; break;
                        case 46: increment = 1099528470784; break;
                        case 47: increment = 1099528470785; break;
                        case 48: increment = 1103806595072; break;
                        case 49: increment = 1103806595073; break;
                        case 50: increment = 1103806595328; break;
                        case 51: increment = 1103806595329; break;
                        case 52: increment = 1103806660608; break;
                        case 53: increment = 1103806660609; break;
                        case 54: increment = 1103806660864; break;
                        case 55: increment = 1103806660865; break;
                        case 56: increment = 1103823372288; break;
                        case 57: increment = 1103823372289; break;
                        case 58: increment = 1103823372544; break;
                        case 59: increment = 1103823372545; break;
                        case 60: increment = 1103823437824; break;
                        case 61: increment = 1103823437825; break;
                        case 62: increment = 1103823438080; break;
                        case 63: increment = 1103823438081; break;
                        case 64: increment = 281474976710656; break;
                        case 65: increment = 281474976710657; break;
                        case 66: increment = 281474976710912; break;
                        case 67: increment = 281474976710913; break;
                        case 68: increment = 281474976776192; break;
                        case 69: increment = 281474976776193; break;
                        case 70: increment = 281474976776448; break;
                        case 71: increment = 281474976776449; break;
                        case 72: increment = 281474993487872; break;
                        case 73: increment = 281474993487873; break;
                        case 74: increment = 281474993488128; break;
                        case 75: increment = 281474993488129; break;
                        case 76: increment = 281474993553408; break;
                        case 77: increment = 281474993553409; break;
                        case 78: increment = 281474993553664; break;
                        case 79: increment = 281474993553665; break;
                        case 80: increment = 281479271677952; break;
                        case 81: increment = 281479271677953; break;
                        case 82: increment = 281479271678208; break;
                        case 83: increment = 281479271678209; break;
                        case 84: increment = 281479271743488; break;
                        case 85: increment = 281479271743489; break;
                        case 86: increment = 281479271743744; break;
                        case 87: increment = 281479271743745; break;
                        case 88: increment = 281479288455168; break;
                        case 89: increment = 281479288455169; break;
                        case 90: increment = 281479288455424; break;
                        case 91: increment = 281479288455425; break;
                        case 92: increment = 281479288520704; break;
                        case 93: increment = 281479288520705; break;
                        case 94: increment = 281479288520960; break;
                        case 95: increment = 281479288520961; break;
                        case 96: increment = 282574488338432; break;
                        case 97: increment = 282574488338433; break;
                        case 98: increment = 282574488338688; break;
                        case 99: increment = 282574488338689; break;
                        case 100: increment = 282574488403968; break;
                        case 101: increment = 282574488403969; break;
                        case 102: increment = 282574488404224; break;
                        case 103: increment = 282574488404225; break;
                        case 104: increment = 282574505115648; break;
                        case 105: increment = 282574505115649; break;
                        case 106: increment = 282574505115904; break;
                        case 107: increment = 282574505115905; break;
                        case 108: increment = 282574505181184; break;
                        case 109: increment = 282574505181185; break;
                        case 110: increment = 282574505181440; break;
                        case 111: increment = 282574505181441; break;
                        case 112: increment = 282578783305728; break;
                        case 113: increment = 282578783305729; break;
                        case 114: increment = 282578783305984; break;
                        case 115: increment = 282578783305985; break;
                        case 116: increment = 282578783371264; break;
                        case 117: increment = 282578783371265; break;
                        case 118: increment = 282578783371520; break;
                        case 119: increment = 282578783371521; break;
                        case 120: increment = 282578800082944; break;
                        case 121: increment = 282578800082945; break;
                        case 122: increment = 282578800083200; break;
                        case 123: increment = 282578800083201; break;
                        case 124: increment = 282578800148480; break;
                        case 125: increment = 282578800148481; break;
                        case 126: increment = 282578800148736; break;
                        case 127: increment = 282578800148737; break;
                        case 128: increment = 72057594037927936; break;
                        case 129: increment = 72057594037927937; break;
                        case 130: increment = 72057594037928192; break;
                        case 131: increment = 72057594037928193; break;
                        case 132: increment = 72057594037993472; break;
                        case 133: increment = 72057594037993473; break;
                        case 134: increment = 72057594037993728; break;
                        case 135: increment = 72057594037993729; break;
                        case 136: increment = 72057594054705152; break;
                        case 137: increment = 72057594054705153; break;
                        case 138: increment = 72057594054705408; break;
                        case 139: increment = 72057594054705409; break;
                        case 140: increment = 72057594054770688; break;
                        case 141: increment = 72057594054770689; break;
                        case 142: increment = 72057594054770944; break;
                        case 143: increment = 72057594054770945; break;
                        case 144: increment = 72057598332895232; break;
                        case 145: increment = 72057598332895233; break;
                        case 146: increment = 72057598332895488; break;
                        case 147: increment = 72057598332895489; break;
                        case 148: increment = 72057598332960768; break;
                        case 149: increment = 72057598332960769; break;
                        case 150: increment = 72057598332961024; break;
                        case 151: increment = 72057598332961025; break;
                        case 152: increment = 72057598349672448; break;
                        case 153: increment = 72057598349672449; break;
                        case 154: increment = 72057598349672704; break;
                        case 155: increment = 72057598349672705; break;
                        case 156: increment = 72057598349737984; break;
                        case 157: increment = 72057598349737985; break;
                        case 158: increment = 72057598349738240; break;
                        case 159: increment = 72057598349738241; break;
                        case 160: increment = 72058693549555712; break;
                        case 161: increment = 72058693549555713; break;
                        case 162: increment = 72058693549555968; break;
                        case 163: increment = 72058693549555969; break;
                        case 164: increment = 72058693549621248; break;
                        case 165: increment = 72058693549621249; break;
                        case 166: increment = 72058693549621504; break;
                        case 167: increment = 72058693549621505; break;
                        case 168: increment = 72058693566332928; break;
                        case 169: increment = 72058693566332929; break;
                        case 170: increment = 72058693566333184; break;
                        case 171: increment = 72058693566333185; break;
                        case 172: increment = 72058693566398464; break;
                        case 173: increment = 72058693566398465; break;
                        case 174: increment = 72058693566398720; break;
                        case 175: increment = 72058693566398721; break;
                        case 176: increment = 72058697844523008; break;
                        case 177: increment = 72058697844523009; break;
                        case 178: increment = 72058697844523264; break;
                        case 179: increment = 72058697844523265; break;
                        case 180: increment = 72058697844588544; break;
                        case 181: increment = 72058697844588545; break;
                        case 182: increment = 72058697844588800; break;
                        case 183: increment = 72058697844588801; break;
                        case 184: increment = 72058697861300224; break;
                        case 185: increment = 72058697861300225; break;
                        case 186: increment = 72058697861300480; break;
                        case 187: increment = 72058697861300481; break;
                        case 188: increment = 72058697861365760; break;
                        case 189: increment = 72058697861365761; break;
                        case 190: increment = 72058697861366016; break;
                        case 191: increment = 72058697861366017; break;
                        case 192: increment = 72339069014638592; break;
                        case 193: increment = 72339069014638593; break;
                        case 194: increment = 72339069014638848; break;
                        case 195: increment = 72339069014638849; break;
                        case 196: increment = 72339069014704128; break;
                        case 197: increment = 72339069014704129; break;
                        case 198: increment = 72339069014704384; break;
                        case 199: increment = 72339069014704385; break;
                        case 200: increment = 72339069031415808; break;
                        case 201: increment = 72339069031415809; break;
                        case 202: increment = 72339069031416064; break;
                        case 203: increment = 72339069031416065; break;
                        case 204: increment = 72339069031481344; break;
                        case 205: increment = 72339069031481345; break;
                        case 206: increment = 72339069031481600; break;
                        case 207: increment = 72339069031481601; break;
                        case 208: increment = 72339073309605888; break;
                        case 209: increment = 72339073309605889; break;
                        case 210: increment = 72339073309606144; break;
                        case 211: increment = 72339073309606145; break;
                        case 212: increment = 72339073309671424; break;
                        case 213: increment = 72339073309671425; break;
                        case 214: increment = 72339073309671680; break;
                        case 215: increment = 72339073309671681; break;
                        case 216: increment = 72339073326383104; break;
                        case 217: increment = 72339073326383105; break;
                        case 218: increment = 72339073326383360; break;
                        case 219: increment = 72339073326383361; break;
                        case 220: increment = 72339073326448640; break;
                        case 221: increment = 72339073326448641; break;
                        case 222: increment = 72339073326448896; break;
                        case 223: increment = 72339073326448897; break;
                        case 224: increment = 72340168526266368; break;
                        case 225: increment = 72340168526266369; break;
                        case 226: increment = 72340168526266624; break;
                        case 227: increment = 72340168526266625; break;
                        case 228: increment = 72340168526331904; break;
                        case 229: increment = 72340168526331905; break;
                        case 230: increment = 72340168526332160; break;
                        case 231: increment = 72340168526332161; break;
                        case 232: increment = 72340168543043584; break;
                        case 233: increment = 72340168543043585; break;
                        case 234: increment = 72340168543043840; break;
                        case 235: increment = 72340168543043841; break;
                        case 236: increment = 72340168543109120; break;
                        case 237: increment = 72340168543109121; break;
                        case 238: increment = 72340168543109376; break;
                        case 239: increment = 72340168543109377; break;
                        case 240: increment = 72340172821233664; break;
                        case 241: increment = 72340172821233665; break;
                        case 242: increment = 72340172821233920; break;
                        case 243: increment = 72340172821233921; break;
                        case 244: increment = 72340172821299200; break;
                        case 245: increment = 72340172821299201; break;
                        case 246: increment = 72340172821299456; break;
                        case 247: increment = 72340172821299457; break;
                        case 248: increment = 72340172838010880; break;
                        case 249: increment = 72340172838010881; break;
                        case 250: increment = 72340172838011136; break;
                        case 251: increment = 72340172838011137; break;
                        case 252: increment = 72340172838076416; break;
                        case 253: increment = 72340172838076417; break;
                        case 254: increment = 72340172838076672; break;
                        case 255: increment = 72340172838076673; break;
                    }
                    *totals_ptr += increment;
                    totals_ptr++;
                }
            }
        }

        //This loop is proportunally smaller as n increases, so this may not be the best
        // place to spend time optimizing.
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (threshold < totals[offset + bit_id])
                    word |= 1UL << bit_id;
            }
            dst[word_id] = word;
        }
    }

    word_t * representative_impl(word_t ** xs, size_t size) {
        word_t * x = zero();

        std::uniform_int_distribution<size_t> gen(0, size - 1);
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                size_t x_id = gen(rng);
                if ((xs[x_id][word_id] >> bit_id) & 1)
                    word |=  1UL << bit_id;
            }
            x[word_id] = word;
        }

        return x;
    }

    word_t * n_representatives_impl(word_t ** xs, size_t size) {
        word_t * x = zero();

        std::uniform_int_distribution<size_t> gen(0, size - 1);
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                size_t x_id = gen(rng);
                word |=  1UL << (xs[x_id][word_id] >> bit_id) & 1;
            }
            x[word_id] = word;
        }

        return x;
    }

    word_t* threshold(word_t ** xs, size_t size, size_t threshold) {
        if (size < UINT8_MAX)
            return generic_gt<uint8_t>(generic_counts<uint8_t>(xs, size), threshold);
        else if (size < UINT16_MAX)
            return generic_gt<uint16_t>(generic_counts<uint16_t>(xs, size), threshold);
        else
            return generic_gt<uint32_t>(generic_counts<uint32_t>(xs, size), threshold);
    }

    void select_into(word_t * cond, word_t * when1, word_t * when0, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = when0[i] ^ (cond[i] & (when0[i] ^ when1[i]));
        }
    }

    void majority3_into(word_t * x, word_t * y, word_t * z, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ((x[i] & y[i]) | (x[i] & z[i]) | (y[i] & z[i]));
        }
    }

    //NOTE: Empirically this is about 30% better than majority3_into with the compiler's
    // best optimizations, but it makes no difference whatsoever in the memory-bound case
    void majority3_into_avx512(word_t * x, word_t * y, word_t * z, word_t * target) {
        const int step = 512 / (8 * sizeof(word_t));
        word_t* x_ptr = x;
        word_t* y_ptr = y;
        word_t* z_ptr = z;
        word_t* target_ptr = target;

        for (size_t i = 0; i < (BITS/512); ++i) {
            __m512i _x = _mm512_load_si512(x_ptr);
            __m512i _y = _mm512_load_si512(y_ptr);
            __m512i _z = _mm512_load_si512(z_ptr);

            __m512i _x_and_y = _mm512_and_epi64(_x, _y);
            __m512i _x_and_z = _mm512_and_epi64(_x, _z);
            __m512i _y_and_z = _mm512_and_epi64(_y, _z);

            __m512i _result = _mm512_or_epi64(_x_and_y, _x_and_z);
            _result = _mm512_or_epi64(_result, _y_and_z);

            _mm512_store_si512(target_ptr, _result);

            x_ptr += step;
            y_ptr += step;
            z_ptr += step;
            target_ptr += step;
        }
    }

    void true_majority(word_t ** xs, size_t size, word_t *dst) {
        switch (size) {
            case 0:
                rand_into(dst);
                return;
            case 1:
                memcpy(dst, xs[0], BYTES);
                return;
            case 2:
                word_t rnd_buf[WORDS];
                rand_into(rnd_buf);
                select_into(rnd_buf, xs[0], xs[1], dst);
                return;
            case 3:
                //GOAT, reorganize this code, add compile-time switch
                // majority3_into(xs[0], xs[1], xs[2], dst);
                majority3_into_avx512(xs[0], xs[1], xs[2], dst);
                return;
            case 4 ... 255:
            //GOAT, 2-level bit-wise special cases are probably good up until 7 or 9 or so, but they
            // get explosive pretty fast.
            //GOAT, Gathering words together from different vectors and counting multiple 1s at a time
            // is probably the winning strategy in the ~10-256 regime, but I'm still working on that
            // algorithm.

                threshold_into_counting_int8(xs, size, size/2, dst);
                return;
            default:
                //Empirically I'm not seeing any difference in perf between using the different sizes of counter
                //variables because there is so much inefficiency in the threshold_into_counting_generic function
                threshold_into_counting_generic<uint32_t>(xs, size, size/2, dst);
                return;
        }
    }

    word_t* representative(word_t ** xs, size_t size) {
        if (size == 0) return rand();
        else if (size == 1) { word_t * r = empty(); memcpy(r, xs[0], BYTES); return r; }
        else if (size == 2) { word_t * r = rand(); select_into(r, xs[0], xs[1], r); return r; }
        else return representative_impl(xs, size);
    }

    void permute_words_into(word_t * x, word_iter_t* word_permutation, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[word_permutation[i]];
        }
    }

    void inverse_permute_words_into(word_t * x, word_iter_t* word_permutation, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[word_permutation[i]] = x[i];
        }
    }

    word_iter_t* rand_word_permutation(uint32_t seed) {
        std::minstd_rand0 perm_rng(seed);

        auto p = (word_iter_t *) malloc(sizeof(word_iter_t)*WORDS);

        for (word_iter_t i = 0; i < WORDS; ++i)
            p[i] = i;

        std::shuffle(p, p + WORDS, perm_rng);

        return p;
    }

    void permute_into(word_t * x, int32_t perm, word_t * target) {
        if (perm == 0) *target = *x;
        else if (perm > 0) permute_words_into(x, rand_word_permutation(perm), target);
        else inverse_permute_words_into(x, rand_word_permutation(-perm), target);
    }

    void rehash_into(word_t * x, word_t * target) {
        TurboSHAKE(512, (uint8_t *)x, BYTES, 0x1F, (uint8_t *)target, BYTES);
    }
}
#endif //BHV_PACKED_H
