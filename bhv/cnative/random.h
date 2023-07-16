

void rand_into(word_t * x) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        x[i] = rng();
    }
}

avx2_pcg32_random_t avx2_key = {
        .state = {_mm256_set_epi64x(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc), _mm256_set_epi64x(0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49)},
        .inc = {_mm256_set_epi64x(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640), _mm256_set_epi64x(0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e)},
        .pcg32_mult_l =  _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffffu),
        .pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32)};

void rand_into_avx2(word_t * x) {
    for (word_iter_t i = 0; i < WORDS; i += 4) {
        _mm256_storeu_si256((__m256i*)(x + i), avx2_pcg32_random_r(&avx2_key));
    }
}


// TODO adopt the same AVX feature control as majority.h and threshold.h

// avx512_pcg32_random_t avx512_key = {
//     .state = _mm512_set_epi64(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc, 0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49),
//     .inc = _mm512_set_epi64(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640, 0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e),
//         .multiplier =  _mm512_set1_epi64(0x5851f42d4c957f2d)};

// void rand_into_avx512(word_t * x) {
//     for (word_iter_t i = 0; i < WORDS; i += 8) {
//         _mm512_storeu_si512((__m512i*)(x + i), avx512_pcg32_random_r(&avx512_key));
//     }
// }

void random_into(word_t * x, float_t p) {
    std::bernoulli_distribution gen(p);

    for (word_iter_t i = 0; i < WORDS; ++i) {
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            if (gen(rng))
                word |= 1UL << bit_id;
        }
        x[i] = word;
    }
}

void random_into_avx512(word_t* dst, float_t p) {

    //Each chunk is 8 bits, so 8 chunks means we'll use up to 64 bits of random for each input
    // output value.  On average, we'll use substantially fewer; ~1.3 chunks each, or ~10 bits
    //
    //Increase this number for more absolute precision.  You likely won't notice any performance
    // increase from making it smaller.
    //
    //ALSO NOTE: The precision of the floating point math costs us enough that there is no point
    // going higher than 8.  In fact, there may be no reason to go higher than 6, using a 64bit
    // float for the threshold.
    const int num_chunks = 6;

    //First start by constructing a map in 8-bit chunks for the threshold float.
    //Coming out of this loop, each chunk can be any value, 0x00 to 0xFF.
    uint8_t chunks[num_chunks];
    __m512i chunk_vecs[num_chunks];
    float_t x = p;
    for (int i=0; i<num_chunks; i++) {
        chunks[i] = 0;
        for (int bit=7; bit>=0; bit--) {
            if (x<0.5) {
                x *= 2.0;
            } else {
                chunks[i] += 1 << bit;
                x -= 0.5;
                x *= 2.0;
            }
        }
        chunk_vecs[i] = _mm512_set1_epi8(chunks[i]);
    }

    // //DEBUG CODE.  GOAT, can be deleted
    // for (int i=0; i<num_chunks; i++) {
    //     std::cout << (long)chunks[i] << ", ";
    // }
    // std::cout << "\n";

    //We'll generate the final output, 64 bits at a time
    for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (8/sizeof(word_t))) {

        uint64_t final_bits = 0;
        uint64_t final_mask = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed
        int chunk_idx = 0;
        while (chunk_idx < num_chunks) {
            //Get 512 bits of true random, which is 8 input bits for each output bit
            //TODO: Get the 512-bit random to actually work
            __m512i true_random;
            ((__m256i*)(&true_random))[0] = avx2_pcg32_random_r(&avx2_key);
            ((__m256i*)(&true_random))[1] = avx2_pcg32_random_r(&avx2_key);

            //We have a definitive answer for every 8-bit random that is above or below the chunk
            // value, but we need to continue to the next chunk if they are equal
            __mmask64 gt_bits = _mm512_cmpgt_epu8_mask(chunk_vecs[chunk_idx], true_random);
            __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(chunk_vecs[chunk_idx], true_random);
            uint64_t chunk_mask = gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
            uint64_t write_mask = final_mask & chunk_mask; //1 for every bit we will write
            final_bits = final_bits | (write_mask & gt_bits);
            final_mask = final_mask & ~write_mask;

            //This should be true ~77.8% of the time.  (255/256)^64
            if (final_mask == 0) {
                //GOAT debug printf only
                // std::cout << chunk_idx << " breakin out!\n";
                break;
            } else {
                //GOAT debug printf only
                // std::cout << chunk_idx << " loopin again, (mask=" << std::hex << final_mask << "), ";
            }

            chunk_idx++;
        }

        *((uint64_t*)(dst + word_offset)) = final_bits;
    }




}



// // TODO include in benchmark.cpp, along with its brother that does |=
// void rand2_into(word_t * target, int8_t pow) {
//     for (word_iter_t i = 0; i < WORDS; ++i) {
//         word_t w = rng();
//         for (int8_t p = 1; p < pow; ++p) {
//             w &= rng();
//         }
//         target[i] = w;
//     }
// }

// // Note This could have an AVX-512 implementation with 512-bit float-level log and floor, and probably and equivalent to generate_canonical
// // HOWEVER, probably not worth it at the very moment
// template <bool additive>
// void sparse_random_switch_into(word_t * x, float_t prob, word_t * target) {
//     double inv_log_not_prob = 1. / std::log(1 - prob);
//     size_t skip_count = std::floor(std::log(std::generate_canonical<float_t, 32>(rng)) * inv_log_not_prob);

//     for (word_iter_t i = 0; i < WORDS; ++i) {
//         word_t word = x[i];
//         while (skip_count < BITS_PER_WORD) {
//             if constexpr (additive)
//                 word |= 1UL << skip_count;
//             else
//                 word &= ~(1UL << skip_count);
//             skip_count += std::floor(std::log(std::generate_canonical<float_t, 32>(rng)) * inv_log_not_prob);
//         }
//         skip_count -= BITS_PER_WORD;
//         target[i] = word;
//     }
// }

// void random_into_1tree_sparse(word_t * x, float_t p) {
//     if (p < .36)
//         return sparse_random_switch_into<true>(ZERO, p, x);
//     else if (p > .64)
//         return sparse_random_switch_into<false>(ONE, 1.f - p, x);
//     else {
//         rand_into(x);
//         if (p <= .5)
//             sparse_random_switch_into<false>(x, 2*(.5f - p), x);
//         else
//             sparse_random_switch_into<true>(x, 2*(p - .5f), x);
//     }
// }

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
