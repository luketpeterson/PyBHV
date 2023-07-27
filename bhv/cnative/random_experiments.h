
//******************************************************************************************
// ID: 8-Bit-Unified-Wavefront-AVX-512
// 8-bit early-out
// DESCRIPTION: 64-bit-wide unified wavefront, 8 random bits tested at a time using AVX-512
// STATUS: Works.  No further work planned.  Perf ~3.3µs
//******************************************************************************************

// void random_into_avx512(word_t* dst, float_t p) {

//     //Each chunk is 8 bits, so 8 chunks means we'll use up to 64 bits of random for each input
//     // output value.  On average, we'll use substantially fewer; ~1.3 chunks each, or ~10 bits
//     //
//     //Increase this number for more absolute precision.  You likely won't notice any performance
//     // increase from making it smaller.
//     //
//     //ALSO NOTE: The precision of the floating point math costs us enough that there is no point
//     // going higher than 8.  In fact, there may be no reason to go higher than 6, using a 64bit
//     // float for the threshold.
//     const int num_chunks = 6;

//     //First start by constructing a map in 8-bit chunks for the threshold float.
//     //Coming out of this loop, each chunk can be any value, 0x00 to 0xFF.
//     uint8_t chunks[num_chunks];
//     __m512i chunk_vecs[num_chunks];
//     float_t x = p;
//     for (int i=0; i<num_chunks; i++) {
//         chunks[i] = 0;
//         for (int bit=7; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 chunks[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         chunk_vecs[i] = _mm512_set1_epi8(chunks[i]);
//     }

//     // //DEBUG CODE.  GOAT, can be deleted
//     // for (int i=0; i<num_chunks; i++) {
//     //     std::cout << (long)chunks[i] << ", ";
//     // }
//     // std::cout << "\n";

//     //We'll generate the final output, 64 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (8/sizeof(word_t))) {

//         uint64_t final_bits = 0;
//         uint64_t final_mask = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed
//         int chunk_idx = 0;
//         while (chunk_idx < num_chunks) {
//             //Get 512 bits of true random, which is 8 input bits for each output bit
//             //TODO: Get the 512-bit random to actually work
//             __m512i true_random;
//             ((__m256i*)(&true_random))[0] = avx512_pcg32_random_r(&key512);
//             ((__m256i*)(&true_random))[1] = avx512_pcg32_random_r(&key512);

//             //We have a definitive answer for every 8-bit random that is above or below the chunk
//             // value, but we need to continue to the next chunk if they are equal
//             __mmask64 gt_bits = _mm512_cmpgt_epu8_mask(chunk_vecs[chunk_idx], true_random);
//             __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(chunk_vecs[chunk_idx], true_random);
//             uint64_t chunk_mask = gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
//             uint64_t write_mask = final_mask & chunk_mask; //1 for every bit we will write
//             final_bits = final_bits | (write_mask & gt_bits);
//             final_mask = final_mask & ~write_mask;

//             //This should be true ~77.8% of the time.  (255/256)^64
//             if (final_mask == 0) {
//                 //GOAT debug printf only
//                 // std::cout << chunk_idx << " breakin out!" << std::hex << final_bits <<"\n";
//                 break;
//             } else {
//                 //GOAT debug printf only
//                 // std::cout << chunk_idx << " loopin again, (mask=" << std::hex << final_mask << "), ";
//             }

//             chunk_idx++;
//         }

//         *((uint64_t*)(dst + word_offset)) = final_bits;
//     }
// }

//******************************************************************************************
// ID: 4-Bit-Unified-Wavefront-AVX-512
// 4-bit early-out
// DESCRIPTION: 64-bit-wide unified wavefront, 4 random bits tested at a time using AVX-512
// STATUS: Works.  No further work planned.  Perf ~3.3µs
// COMMENTS: This isn't any faster than testing 8 bits at a time because fundamantally the number
//  of input randoms is basically the same on account of using a unified wavefront.  The slight
//  gain from using fewer bits is offset by the higher cost of taking more trips through the loop
//******************************************************************************************

// void random_into_avx512(word_t* dst, float_t p) {
//     const int num_chunks = 13;

//     __m512i four_bit_mask = _mm512_set1_epi8(0x0F);

//     //First start by constructing a map in 4-bit chunks for the threshold float.
//     //Coming out of this loop, each chunk can be any value, 0x0 to 0xF.
//     uint8_t chunks[num_chunks];
//     __m512i chunk_vecs[num_chunks];
//     float_t x = p;
//     for (int i=0; i<num_chunks; i++) {
//         chunks[i] = 0;
//         for (int bit=3; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 chunks[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         chunk_vecs[i] = _mm512_set1_epi8(chunks[i]);
//     }

//     //We'll generate the final output, 64 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (8/sizeof(word_t))) {

//         uint64_t final_bits = 0;
//         uint64_t final_mask = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed
//         int chunk_idx = 0;
//         while (chunk_idx < num_chunks) {
//             //Get 256 bits of true random, which is 4 input bits for each output bit
//             __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//             //Spread the random out in a 512-bit AVX vector, so the lower 4 bits of each 8-bit value
//             // are random, while the upper 4 bits are 0
//             __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 4);
//             __m512i dup_random;
//             ((__m256i*)(&dup_random))[0] = rand_source0;
//             ((__m256i*)(&dup_random))[1] = rand_source1;
//             __m512i true_random = _mm512_and_epi64(dup_random, four_bit_mask);

//             //We have a definitive answer for every 4-bit random that is above or below the chunk
//             // value, but we need to continue to the next chunk if they are equal
//             __mmask64 gt_bits = _mm512_cmpgt_epu8_mask(chunk_vecs[chunk_idx], true_random);
//             __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(chunk_vecs[chunk_idx], true_random);
//             uint64_t chunk_mask = gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
//             uint64_t write_mask = final_mask & chunk_mask; //1 for every bit we will write
//             final_bits = final_bits | (write_mask & gt_bits);
//             final_mask = final_mask & ~write_mask;

//             //See if we've converged all the bits, or if we need to loop again
//             if (final_mask == 0) {
//                 //GOAT debug printf only
//                 // std::cout << chunk_idx << " breakin out!" << std::hex << final_bits <<"\n";
//                 break;
//             } else {
//                 //GOAT debug printf only
//                 // std::cout << chunk_idx << " loopin again, (mask=" << std::hex << final_mask << "), ";
//             }

//             chunk_idx++;
//         }

//         *((uint64_t*)(dst + word_offset)) = final_bits;
//     }
// }

//******************************************************************************************
// ID: 4Mop
// 4-bit initial, then on-demand mop-up, 64-output-bits at a time
// DESCRIPTION: Start with 4 random input bits tested at a time using AVX-512, then use scalar
//  instructions to mop up unconverged outputs
// STATUS: Works.  Perf ~2.4µs  Next try AVX-512 the mop-up.  Also try starting with 2 bits
//  instead of 4.
// COMMENTS: This function does a pretty good job conserving input bits.  Timing breakdown:
//  About 1,350ns is spent generating and applying the initial 4 bits / output.
//  about 150ns is spent generating the mop-up bits that we use as input and
//  about 800ns is spent applying the mop-up bits (64 bits at a time).
//******************************************************************************************

// const __m512i FOUR_BIT_MASK = _mm512_set1_epi8(0x0F);

// inline void generate_and_test_layer_avx512(__m512i* layer_threshold_vec, __mmask64* gt_bits, uint64_t* chunk_mask) {

//     //Get 256 bits of true random, which is 4 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out in a 512-bit AVX vector, so the lower 4 bits of each 8-bit value
//     // are random, while the upper 4 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 4);
//     __m512i dup_random;
//     ((__m256i*)(&dup_random))[0] = rand_source0;
//     ((__m256i*)(&dup_random))[1] = rand_source1;
//     __m512i true_random = _mm512_and_epi64(dup_random, FOUR_BIT_MASK);

//     //We have a definitive answer for every 4-bit random that is above or below the threshold value
//     *gt_bits = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random);
//     __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by constructing a map in 4-bit chunks for the threshold float.
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0xF.
//     const int num_layers = 13;
//     uint8_t layer_thresholds[num_layers];
//     __m512i layer_threshold_vecs[num_layers];
//     float_t x = p;
//     for (int i=0; i<num_layers; i++) {
//         layer_thresholds[i] = 0;
//         for (int bit=3; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 layer_thresholds[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         layer_threshold_vecs[i] = _mm512_set1_epi8(layer_thresholds[i]);
//     }

//     //Pre-Generate some randoms for layers 1-n, that we can use to mop-up
//     const int mop_bucket_cnt = 16; //8192 bits / 16 (chance_of_layer_advance) / 64 (bits_per_value) * 2 (safety margin)
//     uint64_t layer_gt_bits[mop_bucket_cnt][num_layers-1];
//     uint64_t layer_chunk_mask[mop_bucket_cnt][num_layers-1];
//     int bucket_cnt[num_layers-1];
//     int bucket_idx[num_layers-1];
//     int max_bucket_idx = 16;
//     for (int i=0; i<num_layers-1; i++) {
//         for (int j=0; j<max_bucket_idx; j++) {
//             generate_and_test_layer_avx512(&layer_threshold_vecs[i+1], &layer_gt_bits[j][i], &layer_chunk_mask[j][i]);
//         }
//         max_bucket_idx = (max_bucket_idx / 16) + 1;
//         bucket_cnt[i] = 0;
//         bucket_idx[i] = 0;
//     }

//     //We'll generate the final output, 64 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (8/sizeof(word_t))) {

//         uint64_t final_bits = 0;
//         uint64_t final_mask = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed

//         __mmask64 gt_bits;
//         uint64_t chunk_mask;
//         generate_and_test_layer_avx512(&layer_threshold_vecs[0], &gt_bits, &chunk_mask);

//         uint64_t write_mask = final_mask & chunk_mask; //1 for every bit we will write
//         final_bits = final_bits | (write_mask & gt_bits);
//         final_mask = final_mask & ~write_mask;

//         //Mop up any unconverged bits (64 bits at a time)
//         for (int layer = 0; layer < num_layers-1; layer++) {

//             if (final_mask == 0) {
//                 break;
//             }

//             //Figure out how many random bits we need and which layer has enough to pull them from
//             int needed_bits = _mm_popcnt_u64(final_mask);

//             //Branchless version of:
//             // if (bucket_cnt[layer] + needed_bits > 64) {
//             //     bucket_cnt[layer] = needed_bits;
//             //     bucket_idx[layer]++;
//             // } else {
//             //     bucket_cnt[layer] += needed_bits;
//             // }
//             int new_cnt = bucket_cnt[layer] + needed_bits;
//             int rollover = new_cnt > 64;
//             bucket_cnt[layer] = (rollover * needed_bits) + ((!rollover) * new_cnt);
//             bucket_idx[layer] += rollover;

//             //Distribute the random bits according to the mask we need
//             uint64_t current_gt_bits = _pdep_u64(layer_gt_bits[bucket_idx[layer]][layer], final_mask);
//             final_bits = final_bits | current_gt_bits;
//             uint64_t current_mask_bits = _pdep_u64(layer_chunk_mask[bucket_idx[layer]][layer], final_mask);
//             final_mask = final_mask & ~current_mask_bits;

//             //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//             layer_chunk_mask[bucket_idx[layer]][layer] = layer_chunk_mask[bucket_idx[layer]][layer] >> needed_bits;
//             layer_gt_bits[bucket_idx[layer]][layer] = layer_gt_bits[bucket_idx[layer]][layer] >> needed_bits;
//         }

//         // //Mop up any unconverged bits (single bit at a time, for reference)
//         // for (int bit=0; bit<64; bit++) {
//         //     if ((final_mask >> bit) & 0x1) {
//         //         for (int layer = 0; layer < num_layers-1; layer++) {

//         //             int bucket_idx = bucket_cnt[layer] / 64;

//         //             final_bits = final_bits | ((layer_gt_bits[bucket_idx][layer] & 0x1) << bit);

//         //             bool should_break = layer_chunk_mask[bucket_idx][layer] & 0x1;

//         //             // //GOAT, This is cheeze to rotate (and reuse) random bits, but avoids the issue of running out of bits
//         //             // layer_chunk_mask[layer] = (layer_chunk_mask[layer] >> 1) | (layer_chunk_mask[layer] << 63);
//         //             // layer_gt_bits[layer] = (layer_gt_bits[layer] >> 1) | (layer_gt_bits[layer] << 63);

//         //             layer_chunk_mask[bucket_idx][layer] = layer_chunk_mask[bucket_idx][layer] >> 1;
//         //             layer_gt_bits[bucket_idx][layer] = layer_gt_bits[bucket_idx][layer] >> 1;

//         //             bucket_cnt[layer]++;

//         //             if (should_break) {
//         //                 break;
//         //             }
//         //         }
//         //     }
//         // }

//         *((uint64_t*)(dst + word_offset)) = final_bits;
//     }

//     // for (int layer=0; layer<num_layers-1; layer++) {
//     //     std::cout << "Layer: " << layer << " consumed " << bucket_cnt[layer] << " bits\n";
//     // }
// }

//******************************************************************************************
// ID: 4Mop-512-Wavefront
// 4-bit initial, then on-demand mop-up, 512-output-bits at a time
// DESCRIPTION: Start with 4 random input bits tested at a time using AVX-512, collect 8
//  blocks of 64 into a 512 bit AVX vec, and then use an AVX-512 mop-up loop
// STATUS: Works.  Perf ~2.65µs  Try starting with 2 bits instead of 4.
// COMMENTS: Perf breakdown: 1,300ns generating and applying the initial 4 bit random.
//  300ns initializing the mop-up bits (could be cut in half, see note)
//  1000ns performing mop-up.
//  So AVX-512 mop-up is actually SLOWER than doing it 64 bits at a time.  The reason is
//  is that all the popcnt & pdep calls dispatch at once, and then we stall waiting for
//  them all to finish.  In the 64-bit pipeline we can hide the pdeps and popcnts under
//  other work.  Additionally a 512-bit wavefront has more chance that a few straggler bits
//  cause additional trips through the loop.  But the lack of a vector PDEP (and popcnt for
//  me) is by far the biggest issue.
//******************************************************************************************

// const __m512i ZERO_VEC = _mm512_set1_epi64(0);
// const __m512i FULL_VEC = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
// const __m512i SIXTY_FOUR_VEC = _mm512_set1_epi64(0x40);
// const __m512i FOUR_BIT_MASK = _mm512_set1_epi8(0x0F);

// inline void generate_and_test_layer_avx512(__m512i* layer_threshold_vec, __mmask64* gt_bits, uint64_t* chunk_mask) {

//     //Get 256 bits of true random, which is 4 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out in a 512-bit AVX vector, so the lower 4 bits of each 8-bit value
//     // are random, while the upper 4 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 4);
//     __m512i dup_random;
//     ((__m256i*)(&dup_random))[0] = rand_source0;
//     ((__m256i*)(&dup_random))[1] = rand_source1;
//     __m512i true_random = _mm512_and_epi64(dup_random, FOUR_BIT_MASK);

//     //We have a definitive answer for every 4-bit random that is above or below the threshold value
//     *gt_bits = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random);
//     __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by constructing a map in 4-bit chunks for the threshold float.
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0xF.
//     const int num_layers = 13;
//     uint8_t layer_thresholds[num_layers];
//     __m512i layer_threshold_vecs[num_layers];
//     float_t x = p;
//     for (int i=0; i<num_layers; i++) {
//         layer_thresholds[i] = 0;
//         for (int bit=3; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 layer_thresholds[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         layer_threshold_vecs[i] = _mm512_set1_epi8(layer_thresholds[i]);
//     }

//     //Pre-Generate some randoms for layers 1-n, that we can use to mop-up
//     const int mop_bucket_cnt = 4; //8192 bits / 16 (chance_of_layer_advance) / 512 (bits_per_value) * 4 (safety margin)
//     __m512i layer_gt_bits[mop_bucket_cnt][num_layers-1];
//     __m512i layer_chunk_mask[mop_bucket_cnt][num_layers-1];
//     __m512i bucket_cnt[num_layers-1];
//     int bucket_idx[num_layers-1];
//     int bucket_init_vec_cnt_table[num_layers-1] = {4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//     for (int i=0; i<num_layers-1; i++) {
//         for (int vec_idx=0; vec_idx<bucket_init_vec_cnt_table[i]; vec_idx++) {
//             for (int el_idx=0; el_idx<8; el_idx++) {
//                 //OPTIMIZATION OPPORTUNITY:  It appears we can save on the order of 150ns by populating
//                 // only the lower bits in the higher levels
//                 generate_and_test_layer_avx512(&layer_threshold_vecs[i+1],
//                     &((uint64_t*)&layer_gt_bits[vec_idx][i])[el_idx],
//                     &((uint64_t*)&layer_chunk_mask[vec_idx][i])[el_idx]);
//             }
//         }
//         bucket_cnt[i] = ZERO_VEC;
//         bucket_idx[i] = 0;
//     }

//     //We'll generate the final output, 512 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (64/sizeof(word_t))) {

//         __m512i final_bits = ZERO_VEC;
//         __m512i final_mask = FULL_VEC; //1 for every bit whose value is still needed

//         __m512i gt_bits;
//         __m512i chunk_mask;
//         for (int el_idx=0; el_idx<8; el_idx++) {
//             generate_and_test_layer_avx512(&layer_threshold_vecs[0],
//                 &((uint64_t*)&gt_bits)[el_idx],
//                 &((uint64_t*)&chunk_mask)[el_idx]);
//         }

//         __m512i write_mask = final_mask & chunk_mask; //1 for every bit we will write
//         final_bits = final_bits | (write_mask & gt_bits);
//         final_mask = final_mask & ~write_mask;

//         //Mop up any unconverged bits (512 bits at a time)
//         for (int layer = 0; layer < num_layers-1; layer++) {

//             //if (final_mask == 0) {
//             //GOAT, there's gotta be a way to check every lane against zero cheaper than this???
//             if (_mm512_cmpeq_epi64_mask(final_mask, ZERO_VEC) == 0xFF) {
//                 break;
//             }

//             //Figure out how many random bits we need and which layer has enough to pull them from
//             //GOAT, this instruction is emulated on my machine so it's no faster than running
//             // _mm_popcnt_u64 8 times, with all the moves to get everything in the right place
//             __m512i needed_bits = _mm512_popcnt_epi64(final_mask);

//             // if (bucket_cnt[layer] + needed_bits > 64) {
//             //     bucket_cnt[layer] = needed_bits;
//             //     bucket_idx[layer]++;
//             // } else {
//             //     bucket_cnt[layer] += needed_bits;
//             // }
//             __m512i new_cnt = _mm512_add_epi64(bucket_cnt[layer], needed_bits);
//             int rollover = _mm512_cmpgt_epi64_mask(new_cnt, SIXTY_FOUR_VEC) != 0;
//             bucket_cnt[layer] = (rollover * needed_bits) + ((!rollover) * new_cnt);
//             bucket_idx[layer] += rollover;

//             //Distribute the random bits according to the mask we need
//             __m512i cur_layer_gt_bits = layer_gt_bits[bucket_idx[layer]][layer];
//             __m512i current_gt_bits;
//             ((uint64_t*)&current_gt_bits)[0] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[0], ((uint64_t*)&final_mask)[0]);
//             ((uint64_t*)&current_gt_bits)[1] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[1], ((uint64_t*)&final_mask)[1]);
//             ((uint64_t*)&current_gt_bits)[2] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[2], ((uint64_t*)&final_mask)[2]);
//             ((uint64_t*)&current_gt_bits)[3] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[3], ((uint64_t*)&final_mask)[3]);
//             ((uint64_t*)&current_gt_bits)[4] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[4], ((uint64_t*)&final_mask)[4]);
//             ((uint64_t*)&current_gt_bits)[5] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[5], ((uint64_t*)&final_mask)[5]);
//             ((uint64_t*)&current_gt_bits)[6] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[6], ((uint64_t*)&final_mask)[6]);
//             ((uint64_t*)&current_gt_bits)[7] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[7], ((uint64_t*)&final_mask)[7]);
//             final_bits = final_bits | current_gt_bits;

//             __m512i cur_layer_chunk = layer_chunk_mask[bucket_idx[layer]][layer];
//             __m512i current_mask_bits;
//             ((uint64_t*)&current_mask_bits)[0] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[0], ((uint64_t*)&final_mask)[0]);
//             ((uint64_t*)&current_mask_bits)[1] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[1], ((uint64_t*)&final_mask)[1]);
//             ((uint64_t*)&current_mask_bits)[2] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[2], ((uint64_t*)&final_mask)[2]);
//             ((uint64_t*)&current_mask_bits)[3] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[3], ((uint64_t*)&final_mask)[3]);
//             ((uint64_t*)&current_mask_bits)[4] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[4], ((uint64_t*)&final_mask)[4]);
//             ((uint64_t*)&current_mask_bits)[5] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[5], ((uint64_t*)&final_mask)[5]);
//             ((uint64_t*)&current_mask_bits)[6] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[6], ((uint64_t*)&final_mask)[6]);
//             ((uint64_t*)&current_mask_bits)[7] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[7], ((uint64_t*)&final_mask)[7]);
//             final_mask = final_mask & ~current_mask_bits;

//             //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//             layer_chunk_mask[bucket_idx[layer]][layer] = layer_chunk_mask[bucket_idx[layer]][layer] >> needed_bits;
//             layer_gt_bits[bucket_idx[layer]][layer] = layer_gt_bits[bucket_idx[layer]][layer] >> needed_bits;
//         }

//         *((__m512i*)(dst + word_offset)) = final_bits;
//     }

//     // for (int layer=0; layer<num_layers-1; layer++) {
//     //     uint64_t* bucket_cnt_ptr = (uint64_t*)&bucket_cnt[layer];
//     //     std::cout << "Layer: " << layer << " consumed= " << bucket_cnt_ptr[7] << " idx= " << bucket_idx[layer] << "\n";
//     // }
// }


//******************************************************************************************
// ID: 4Mop-256Wavefront-AVX
// 4-bit initial, then on-demand mop-up, 256-output-bits at a time
// DESCRIPTION: Start with 4 random input bits tested at a time using AVX-512, collect 4
//  blocks of 64 into a 256 bit AVX vec, and then use an AVX-256 mop-up loop
// STATUS: Works.  Perf ~2.75µs  Try starting with 2 bits instead of 4.
// COMMENTS: Perf breakdown: 1,300ns generating and applying the initial 4 bit random.
//  200ns initializing the mop-up bits
//  1200ns performing mop-up.
//  AVX-256 mop-up is the slowest of all!  I guess it's the worst of both worlds between
//  blocking on pdep and popcnt, and also looping 2x more vs. the AVX-512 version.
//******************************************************************************************

// const __m256i ZERO_VEC = _mm256_set1_epi64x(0);
// const __m256i FULL_VEC = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF);
// const __m256i SIXTY_FOUR_VEC = _mm256_set1_epi64x(0x40);
// const __m512i FOUR_BIT_MASK = _mm512_set1_epi8(0x0F);

// inline void generate_and_test_layer_avx512(__m512i* layer_threshold_vec, __mmask64* gt_bits, uint64_t* chunk_mask) {

//     //Get 256 bits of true random, which is 4 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out in a 512-bit AVX vector, so the lower 4 bits of each 8-bit value
//     // are random, while the upper 4 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 4);
//     __m512i dup_random;
//     ((__m256i*)(&dup_random))[0] = rand_source0;
//     ((__m256i*)(&dup_random))[1] = rand_source1;
//     __m512i true_random = _mm512_and_epi64(dup_random, FOUR_BIT_MASK);

//     //We have a definitive answer for every 4-bit random that is above or below the threshold value
//     *gt_bits = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random);
//     __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by constructing a map in 4-bit chunks for the threshold float.
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0xF.
//     const int num_layers = 13;
//     uint8_t layer_thresholds[num_layers];
//     __m512i layer_threshold_vecs[num_layers];
//     float_t x = p;
//     for (int i=0; i<num_layers; i++) {
//         layer_thresholds[i] = 0;
//         for (int bit=3; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 layer_thresholds[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         layer_threshold_vecs[i] = _mm512_set1_epi8(layer_thresholds[i]);
//     }

//     //Pre-Generate some randoms for layers 1-n, that we can use to mop-up
//     const int mop_bucket_cnt = 6; //8192 bits / 16 (chance_of_layer_advance) / 256 (bits_per_value) * 3 (safety margin)
//     __m256i layer_gt_bits[mop_bucket_cnt][num_layers-1];
//     __m256i layer_chunk_mask[mop_bucket_cnt][num_layers-1];
//     __m256i bucket_cnt[num_layers-1];
//     int bucket_idx[num_layers-1];
//     int bucket_init_vec_cnt_table[num_layers-1] = {6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//     for (int i=0; i<num_layers-1; i++) {
//         for (int vec_idx=0; vec_idx<bucket_init_vec_cnt_table[i]; vec_idx++) {
//             for (int el_idx=0; el_idx<4; el_idx++) {
//                 generate_and_test_layer_avx512(&layer_threshold_vecs[i+1],
//                     &((uint64_t*)&layer_gt_bits[vec_idx][i])[el_idx],
//                     &((uint64_t*)&layer_chunk_mask[vec_idx][i])[el_idx]);
//             }
//         }
//         bucket_cnt[i] = ZERO_VEC;
//         bucket_idx[i] = 0;
//     }

//     //We'll generate the final output, 256 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (32/sizeof(word_t))) {

//         __m256i final_bits = ZERO_VEC;
//         __m256i final_mask = FULL_VEC; //1 for every bit whose value is still needed

//         __m256i gt_bits;
//         __m256i chunk_mask;
//         for (int el_idx=0; el_idx<4; el_idx++) {
//             generate_and_test_layer_avx512(&layer_threshold_vecs[0],
//                 &((uint64_t*)&gt_bits)[el_idx],
//                 &((uint64_t*)&chunk_mask)[el_idx]);
//         }

//         __m256i write_mask = final_mask & chunk_mask; //1 for every bit we will write
//         final_bits = final_bits | (write_mask & gt_bits);
//         final_mask = final_mask & ~write_mask;

//         //Mop up any unconverged bits (256 bits at a time)
//         for (int layer = 0; layer < num_layers-1; layer++) {

//             //if (final_mask == 0) {
//             if (_mm256_testz_si256(final_mask, final_mask)) {
//                 break;
//             }

//             //Figure out how many random bits we need and which layer has enough to pull them from
//             //GOAT, this instruction is emulated on my machine so it's no faster than running
//             // _mm_popcnt_u64 4 times, with all the moves to get everything in the right place
//             __m256i needed_bits = _mm256_popcnt_epi64(final_mask);

//             // if (bucket_cnt[layer] + needed_bits > 64) {
//             //     bucket_cnt[layer] = needed_bits;
//             //     bucket_idx[layer]++;
//             // } else {
//             //     bucket_cnt[layer] += needed_bits;
//             // }
//             __m256i new_cnt = _mm256_add_epi64(bucket_cnt[layer], needed_bits);
//             __m256i shifted = _mm256_srai_epi64(new_cnt, 6); //Test bit 7
//             int rollover = _mm256_testz_si256(shifted, shifted) == 0;
//             bucket_cnt[layer] = (rollover * needed_bits) + ((!rollover) * new_cnt);
//             bucket_idx[layer] += rollover;

//             //Distribute the random bits according to the mask we need
//             __m256i cur_layer_gt_bits = layer_gt_bits[bucket_idx[layer]][layer];
//             __m256i current_gt_bits;
//             ((uint64_t*)&current_gt_bits)[0] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[0], ((uint64_t*)&final_mask)[0]);
//             ((uint64_t*)&current_gt_bits)[1] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[1], ((uint64_t*)&final_mask)[1]);
//             ((uint64_t*)&current_gt_bits)[2] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[2], ((uint64_t*)&final_mask)[2]);
//             ((uint64_t*)&current_gt_bits)[3] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[3], ((uint64_t*)&final_mask)[3]);
//             final_bits = final_bits | current_gt_bits;

//             __m256i cur_layer_chunk = layer_chunk_mask[bucket_idx[layer]][layer];
//             __m256i current_mask_bits;
//             ((uint64_t*)&current_mask_bits)[0] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[0], ((uint64_t*)&final_mask)[0]);
//             ((uint64_t*)&current_mask_bits)[1] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[1], ((uint64_t*)&final_mask)[1]);
//             ((uint64_t*)&current_mask_bits)[2] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[2], ((uint64_t*)&final_mask)[2]);
//             ((uint64_t*)&current_mask_bits)[3] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[3], ((uint64_t*)&final_mask)[3]);
//             final_mask = final_mask & ~current_mask_bits;

//             //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//             layer_chunk_mask[bucket_idx[layer]][layer] = layer_chunk_mask[bucket_idx[layer]][layer] >> needed_bits;
//             layer_gt_bits[bucket_idx[layer]][layer] = layer_gt_bits[bucket_idx[layer]][layer] >> needed_bits;
//         }

//         *((__m256i*)(dst + word_offset)) = final_bits;
//     }
// }

//******************************************************************************************
// ID: 2Mop-128Wavefront-SSE
// 2-bit initial, then on-demand mop-up, 128-output-bits at a time.  SSE
// DESCRIPTION: Start with 2 random input bits tested at a time using AVX-512, then use SSE
//  instructions to mop up unconverged outputs
// STATUS: Works.  Perf ~4.4µs.  Unacceptable!!!!!
// COMMENTS: Perf breakdown: 700ns generating and applying the initial 2 bit random.
//  350ns initializing the mop-up bits, and then 3,350ns mopping up!!!!!!!!
//  I knew the mop up would cost more here, because each layer converges 75% of the bits as
//  opposed to 93% with the 4-bit threshold.  This means that a typical 128-bit chunk takes
//  4-5 layers to converge, vs. 1-2 layers using a 4-bit threshold.  But the horrible perf
//  we are seeing can't be explained by just that.  My new theory is that the 64-bit mop-up
//  loop is the best performing one because it gives autovectorization a chance.  So, next
//  I will try building a 128 bit chunk from all 64-bit operations, giving the
//  autovectorizer more room to maneuver. 
//******************************************************************************************

// const __m128i ZERO_VEC = _mm_set1_epi64x(0);
// const __m128i FULL_VEC = _mm_set1_epi64x(0xFFFFFFFFFFFFFFFF);

// const __m512i TWO_BIT_MASK = _mm512_set1_epi8(0x03);

// inline void generate_and_test_layer_avx512(__m512i* layer_threshold_vec, __m128i* gt_bits, __m128i* chunk_mask) {

//     //Get 256 bits of true random, which is 2 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out into two 512-bit AVX vectors, so the lower 2 bits of each 8-bit value
//     // are random, while the upper 6 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 2);
//     __m512i dup_random0;
//     ((__m256i*)(&dup_random0))[0] = rand_source0;
//     ((__m256i*)(&dup_random0))[1] = rand_source1;
//     __m512i true_random0 = _mm512_and_epi64(dup_random0, TWO_BIT_MASK);

//     __m256i rand_source2 = _mm256_srli_epi16(rand_source0, 4);
//     __m256i rand_source3 = _mm256_srli_epi16(rand_source0, 6);
//     __m512i dup_random1;
//     ((__m256i*)(&dup_random1))[0] = rand_source2;
//     ((__m256i*)(&dup_random1))[1] = rand_source3;
//     __m512i true_random1 = _mm512_and_epi64(dup_random1, TWO_BIT_MASK);

//     //We have a definitive answer for every 4-bit random that is above or below the threshold value
//     __m128i eq_bits;
//     ((uint64_t*)gt_bits)[0] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)&eq_bits)[0] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)gt_bits)[1] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random1);
//     ((uint64_t*)&eq_bits)[1] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random1);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// //4-bit early-out, on-demand mop-up
// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by constructing a map in 2-bit chunks for the threshold float.
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0x3.
//     const int num_layers = 26;
//     uint8_t layer_thresholds[num_layers];
//     __m512i layer_threshold_vecs[num_layers];
//     float_t x = p;
//     for (int i=0; i<num_layers; i++) {
//         layer_thresholds[i] = 0;
//         for (int bit=1; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 layer_thresholds[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         layer_threshold_vecs[i] = _mm512_set1_epi8(layer_thresholds[i]);
//     }

//     //Pre-Generate some randoms for layers 1-n, that we can use to mop-up
//     const int mop_bucket_cnt = 32; //8192 bits / 4 (chance_of_layer_advance) / 128 (bits_per_value) * 2 (safety margin)
//     __m128i layer_gt_bits[mop_bucket_cnt][num_layers-1];
//     __m128i layer_chunk_mask[mop_bucket_cnt][num_layers-1];
//     __m128i bucket_cnt[num_layers-1];
//     int bucket_idx[num_layers-1];
//     int bucket_init_vec_cnt_table[num_layers-1] = {32, 8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//     for (int i=0; i<num_layers-1; i++) {
//         for (int vec_idx=0; vec_idx<bucket_init_vec_cnt_table[i]; vec_idx++) {
//             generate_and_test_layer_avx512(&layer_threshold_vecs[i+1],
//                 &layer_gt_bits[vec_idx][i],
//                 &layer_chunk_mask[vec_idx][i]);
//         }
//         bucket_cnt[i] = ZERO_VEC;
//         bucket_idx[i] = 0;
//     }

//     //We'll generate the final output, 128 bits (16 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (16/sizeof(word_t))) {

//         __m128i final_bits = ZERO_VEC;
//         __m128i final_mask = FULL_VEC; //1 for every bit whose value is still needed

//         __m128i gt_bits;
//         __m128i chunk_mask;
//         generate_and_test_layer_avx512(&layer_threshold_vecs[0], &gt_bits, &chunk_mask);

//         __m128i write_mask = final_mask & chunk_mask; //1 for every bit we will write
//         final_bits = final_bits | (write_mask & gt_bits);
//         final_mask = final_mask & ~write_mask;

//         //Mop up any unconverged bits (128 bits at a time)
//         for (int layer = 0; layer < num_layers-1; layer++) {

//             //if (final_mask == 0) {
//             if (_mm_testz_si128(final_mask, final_mask)) {
//                 break;
//             }

//             //Figure out how many random bits we need and which layer has enough to pull them from
//             //GOAT, this instruction is emulated on my machine so it's no faster than running
//             // _mm_popcnt_u64 twice, with all the moves to get everything in the right place
//             __m128i needed_bits = _mm_popcnt_epi64(final_mask);

//             // if (bucket_cnt[layer] + needed_bits > 64) {
//             //     bucket_cnt[layer] = needed_bits;
//             //     bucket_idx[layer]++;
//             // } else {
//             //     bucket_cnt[layer] += needed_bits;
//             // }
//             __m128i new_cnt = _mm_add_epi64(bucket_cnt[layer], needed_bits);
//             __m128i shifted = _mm_srai_epi64(new_cnt, 6); //Test bit 7
//             int rollover = _mm_testz_si128(shifted, shifted) == 0;
//             bucket_cnt[layer] = (rollover * needed_bits) + ((!rollover) * new_cnt);
//             bucket_idx[layer] += rollover;

//             //Distribute the random bits according to the mask we need
//             __m128i cur_layer_gt_bits = layer_gt_bits[bucket_idx[layer]][layer];
//             __m128i current_gt_bits;
//             ((uint64_t*)&current_gt_bits)[0] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[0], ((uint64_t*)&final_mask)[0]);
//             ((uint64_t*)&current_gt_bits)[1] = _pdep_u64(((uint64_t*)&cur_layer_gt_bits)[1], ((uint64_t*)&final_mask)[1]);
//             final_bits = final_bits | current_gt_bits;

//             __m128i cur_layer_chunk = layer_chunk_mask[bucket_idx[layer]][layer];
//             __m128i current_mask_bits;
//             ((uint64_t*)&current_mask_bits)[0] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[0], ((uint64_t*)&final_mask)[0]);
//             ((uint64_t*)&current_mask_bits)[1] = _pdep_u64(((uint64_t*)&cur_layer_chunk)[1], ((uint64_t*)&final_mask)[1]);
//             final_mask = final_mask & ~current_mask_bits;

//             //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//             layer_chunk_mask[bucket_idx[layer]][layer] = layer_chunk_mask[bucket_idx[layer]][layer] >> needed_bits;
//             layer_gt_bits[bucket_idx[layer]][layer] = layer_gt_bits[bucket_idx[layer]][layer] >> needed_bits;
//         }

//         *((__m128i*)(dst + word_offset)) = final_bits;
//     }

//     // for (int layer=0; layer<num_layers-1; layer++) {
//     //     std::cout << "Layer: " << layer << " consumed " << bucket_cnt[layer] << " bits\n";
//     // }
// }

//******************************************************************************************
// ID: 2Mop-128Wavefront-DualScalar
// 2-bit initial, then on-demand mop-up, 128-output-bits at a time, using scalar instructions
// DESCRIPTION: Start with 2 random input bits tested at a time using AVX-512, then use
//  overlapping scalar instruction paths to mop up unconverged outputs
// STATUS: Works.  Perf ~3.8µs.  Still Unacceptable!!!!!
// COMMENTS: Perf breakdown: 700ns generating and applying the initial 2 bit random.
//  350ns initializing the mop-up bits, and then 2,750ns mopping up!!!!!!!!
//  Letting the auto-vectorizer do its thing saved ~20%.  Which is substantial, but nowhere
//  near enough to claw back what we lost.  It seems more trips through the mop-up loop have
//  a worse than linear cost associated with them.
//  So next, I will try a 2-bit initial filter pass, followed by a 4-bit mop-up loop
//******************************************************************************************

// const __m512i TWO_BIT_MASK = _mm512_set1_epi8(0x03);

// inline void generate_and_test_layer_avx512(__m512i* layer_threshold_vec, __m128i* gt_bits, __m128i* chunk_mask) {

//     //Get 256 bits of true random, which is 2 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out into two 512-bit AVX vectors, so the lower 2 bits of each 8-bit value
//     // are random, while the upper 6 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 2);
//     __m512i dup_random0;
//     ((__m256i*)(&dup_random0))[0] = rand_source0;
//     ((__m256i*)(&dup_random0))[1] = rand_source1;
//     __m512i true_random0 = _mm512_and_epi64(dup_random0, TWO_BIT_MASK);

//     __m256i rand_source2 = _mm256_srli_epi16(rand_source0, 4);
//     __m256i rand_source3 = _mm256_srli_epi16(rand_source0, 6);
//     __m512i dup_random1;
//     ((__m256i*)(&dup_random1))[0] = rand_source2;
//     ((__m256i*)(&dup_random1))[1] = rand_source3;
//     __m512i true_random1 = _mm512_and_epi64(dup_random1, TWO_BIT_MASK);

//     //We have a definitive answer for every 4-bit random that is above or below the threshold value
//     __m128i eq_bits;
//     ((uint64_t*)gt_bits)[0] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)&eq_bits)[0] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)gt_bits)[1] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random1);
//     ((uint64_t*)&eq_bits)[1] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random1);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by constructing a map in 2-bit chunks for the threshold float.
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0x3.
//     const int num_layers = 26;
//     uint8_t layer_thresholds[num_layers];
//     __m512i layer_threshold_vecs[num_layers];
//     float_t x = p;
//     for (int i=0; i<num_layers; i++) {
//         layer_thresholds[i] = 0;
//         for (int bit=1; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 layer_thresholds[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         layer_threshold_vecs[i] = _mm512_set1_epi8(layer_thresholds[i]);
//     }

//     //Pre-Generate some randoms for layers 1-n, that we can use to mop-up
//     const int mop_bucket_cnt = 32; //8192 bits / 4 (chance_of_layer_advance) / 128 (bits_per_value) * 2 (safety margin)
//     uint64_t layer_gt_bits0[mop_bucket_cnt][num_layers-1];
//     uint64_t layer_gt_bits1[mop_bucket_cnt][num_layers-1];
//     uint64_t layer_chunk_mask0[mop_bucket_cnt][num_layers-1];
//     uint64_t layer_chunk_mask1[mop_bucket_cnt][num_layers-1];
//     uint64_t bucket_cnt0[num_layers-1];
//     uint64_t bucket_cnt1[num_layers-1];
//     int bucket_idx[num_layers-1];
//     int bucket_init_vec_cnt_table[num_layers-1] = {32, 8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//     for (int i=0; i<num_layers-1; i++) {
//         for (int vec_idx=0; vec_idx<bucket_init_vec_cnt_table[i]; vec_idx++) {
//             uint64_t gt_bits[2];
//             uint64_t chunk_mask[2];
//             generate_and_test_layer_avx512(&layer_threshold_vecs[i+1],
//                 (__m128i*)gt_bits, (__m128i*)chunk_mask);
//             layer_gt_bits0[vec_idx][i] = gt_bits[0];
//             layer_gt_bits1[vec_idx][i] = gt_bits[1];
//             layer_chunk_mask0[vec_idx][i] = chunk_mask[0];
//             layer_chunk_mask1[vec_idx][i] = chunk_mask[1];
//         }
//         bucket_cnt0[i] = 0;
//         bucket_cnt1[i] = 0;
//         bucket_idx[i] = 0;
//     }

//     //We'll generate the final output, 128 bits (16 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (16/sizeof(word_t))) {

//         uint64_t final_bits0 = 0;
//         uint64_t final_bits1 = 0;
//         uint64_t final_mask0 = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed
//         uint64_t final_mask1 = 0xFFFFFFFFFFFFFFFF;

//         uint64_t gt_bits[2];
//         uint64_t chunk_mask[2];
//         generate_and_test_layer_avx512(&layer_threshold_vecs[0], (__m128i*)gt_bits, (__m128i*)chunk_mask);

//         uint64_t write_mask0 = final_mask0 & chunk_mask[0]; //1 for every bit we will write
//         uint64_t write_mask1 = final_mask1 & chunk_mask[1];
//         final_bits0 = final_bits0 | (write_mask0 & gt_bits[0]);
//         final_bits1 = final_bits1 | (write_mask1 & gt_bits[1]);
//         final_mask0 = final_mask0 & ~write_mask0;
//         final_mask1 = final_mask1 & ~write_mask1;

//         //Mop up any unconverged bits (128 bits at a time)
//         for (int layer = 0; layer < num_layers-1; layer++) {

//             if (final_mask0 == 0 && final_mask1 == 0) {
//                 break;
//             }

//             //Figure out how many random bits we need and which layer has enough to pull them from
//             int needed_bits0 = _mm_popcnt_u64(final_mask0);
//             int needed_bits1 = _mm_popcnt_u64(final_mask1);

//             //Branchless version of:
//             // if (bucket_cnt[layer] + needed_bits > 64) {
//             //     bucket_cnt[layer] = needed_bits;
//             //     bucket_idx[layer]++;
//             // } else {
//             //     bucket_cnt[layer] += needed_bits;
//             // }
//             int new_cnt0 = bucket_cnt0[layer] + needed_bits0;
//             int new_cnt1 = bucket_cnt1[layer] + needed_bits1;
//             int rollover = new_cnt0 > 64 || new_cnt1 > 64;
//             bucket_cnt0[layer] = (rollover * needed_bits0) + ((!rollover) * new_cnt0);
//             bucket_cnt1[layer] = (rollover * needed_bits1) + ((!rollover) * new_cnt1);
//             bucket_idx[layer] += rollover;

//             //Distribute the random bits according to the mask we need
//             uint64_t current_gt_bits0 = _pdep_u64(layer_gt_bits0[bucket_idx[layer]][layer], final_mask0);
//             uint64_t current_gt_bits1 = _pdep_u64(layer_gt_bits1[bucket_idx[layer]][layer], final_mask1);
//             final_bits0 = final_bits0 | current_gt_bits0;
//             final_bits1 = final_bits1 | current_gt_bits1;
//             uint64_t current_mask_bits0 = _pdep_u64(layer_chunk_mask0[bucket_idx[layer]][layer], final_mask0);
//             uint64_t current_mask_bits1 = _pdep_u64(layer_chunk_mask1[bucket_idx[layer]][layer], final_mask1);
//             final_mask0 = final_mask0 & ~current_mask_bits0;
//             final_mask1 = final_mask1 & ~current_mask_bits1;

//             //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//             layer_chunk_mask0[bucket_idx[layer]][layer] = layer_chunk_mask0[bucket_idx[layer]][layer] >> needed_bits0;
//             layer_chunk_mask1[bucket_idx[layer]][layer] = layer_chunk_mask1[bucket_idx[layer]][layer] >> needed_bits1;
//             layer_gt_bits0[bucket_idx[layer]][layer] = layer_gt_bits0[bucket_idx[layer]][layer] >> needed_bits0;
//             layer_gt_bits1[bucket_idx[layer]][layer] = layer_gt_bits1[bucket_idx[layer]][layer] >> needed_bits1;
//         }

//         // *(dst + word_offset) = final_bits0;
//         // *(dst + word_offset+1) = final_bits1;
//         *(dst + word_offset) = final_bits0;
//         *(dst + word_offset+1) = final_bits1;
//     }
// }

//******************************************************************************************
// ID: 2-4Mop-128Wavefront-DualScalar
// 2-bit initial, then on-demand mop-up using 4 bits of input, 128-output-bits at a time.
// DESCRIPTION: Start with 2 random input bits tested at a time using AVX-512, then use
//  scalar instructions, doubled up, to do the mop up, with 4-bit thresholds
// STATUS: Works.  Perf ~2.85µs
// COMMENTS: Perf breakdown: 600ns generating and applying the initial 2 bit random.
//  650ns initializing the mop-up bits (we need more because the primary filter doesn't catch as much)
//  1600ns performing the mop-up.
//******************************************************************************************

// const __m512i TWO_BIT_MASK = _mm512_set1_epi8(0x03);
// const __m512i FOUR_BIT_MASK = _mm512_set1_epi8(0x0F);

// inline void generate_and_test_2bit_layer_avx512(__m512i* layer_threshold_vec, __m128i* gt_bits, __m128i* chunk_mask) {

//     //Get 256 bits of true random, which is 2 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out into two 512-bit AVX vectors, so the lower 2 bits of each 8-bit value
//     // are random, while the upper 6 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 2);
//     __m512i dup_random0;
//     ((__m256i*)(&dup_random0))[0] = rand_source0;
//     ((__m256i*)(&dup_random0))[1] = rand_source1;
//     __m512i true_random0 = _mm512_and_epi64(dup_random0, TWO_BIT_MASK);

//     __m256i rand_source2 = _mm256_srli_epi16(rand_source0, 4);
//     __m256i rand_source3 = _mm256_srli_epi16(rand_source0, 6);
//     __m512i dup_random1;
//     ((__m256i*)(&dup_random1))[0] = rand_source2;
//     ((__m256i*)(&dup_random1))[1] = rand_source3;
//     __m512i true_random1 = _mm512_and_epi64(dup_random1, TWO_BIT_MASK);

//     //We have a definitive answer for every 2-bit random that is above or below the threshold value
//     __m128i eq_bits;
//     ((uint64_t*)gt_bits)[0] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)&eq_bits)[0] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)gt_bits)[1] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random1);
//     ((uint64_t*)&eq_bits)[1] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random1);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// inline void generate_and_test_4bit_layer_avx512(__m512i* layer_threshold_vec, __mmask64* gt_bits, uint64_t* chunk_mask) {

//     //Get 256 bits of true random, which is 4 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out in a 512-bit AVX vector, so the lower 4 bits of each 8-bit value
//     // are random, while the upper 4 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 4);
//     __m512i dup_random;
//     ((__m256i*)(&dup_random))[0] = rand_source0;
//     ((__m256i*)(&dup_random))[1] = rand_source1;
//     __m512i true_random = _mm512_and_epi64(dup_random, FOUR_BIT_MASK);

//     //We have a definitive answer for every 4-bit random that is above or below the threshold value
//     *gt_bits = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random);
//     __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by finding a 2-bit threshold from the float, to rule out 75% of the bit patterns.
//     uint8_t primary_threshold = 0;
//     __m512i primary_threshold_vec;
//     float_t x = p;
//     for (int bit=1; bit>=0; bit--) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             primary_threshold += 1 << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }
//     primary_threshold_vec = _mm512_set1_epi8(primary_threshold);

//     //Next, initialize 4-bit chunks for the remaining precision we want from the float threshold
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0xF.
//     const int num_layers = 12;
//     uint8_t layer_thresholds[num_layers];
//     __m512i layer_threshold_vecs[num_layers];
//     for (int i=0; i<num_layers; i++) {
//         layer_thresholds[i] = 0;
//         for (int bit=3; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 layer_thresholds[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         layer_threshold_vecs[i] = _mm512_set1_epi8(layer_thresholds[i]);
//     }

//     //Pre-Generate some randoms for the mop-up layers
//     const int mop_bucket_cnt = 32; //8192 bits / 4 (primary screen) / 128 (bits_per_value) * 2 (safety margin)
//     uint64_t layer_gt_bits0[mop_bucket_cnt][num_layers];
//     uint64_t layer_gt_bits1[mop_bucket_cnt][num_layers];
//     uint64_t layer_chunk_mask0[mop_bucket_cnt][num_layers];
//     uint64_t layer_chunk_mask1[mop_bucket_cnt][num_layers];
//     uint64_t bucket_cnt0[num_layers];
//     uint64_t bucket_cnt1[num_layers];
//     int bucket_idx[num_layers];
//     int bucket_init_vec_cnt_table[num_layers] = {32, 8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//     for (int i=0; i<num_layers; i++) {
//         for (int vec_idx=0; vec_idx<bucket_init_vec_cnt_table[i]; vec_idx++) {
//             generate_and_test_4bit_layer_avx512(&layer_threshold_vecs[i],
//                 &layer_gt_bits0[vec_idx][i],
//                 &layer_chunk_mask0[vec_idx][i]);
//             generate_and_test_4bit_layer_avx512(&layer_threshold_vecs[i],
//                 &layer_gt_bits1[vec_idx][i],
//                 &layer_chunk_mask1[vec_idx][i]);
//         }
//         bucket_cnt0[i] = 0;
//         bucket_cnt1[i] = 0;
//         bucket_idx[i] = 0;
//     }

//     // //Debug perf stats
//     // int layer_n_passes[num_layers] = {0, 0, 0, 0, 0};
//     // int tot_needed_bits_per_layer[num_layers] = {0, 0, 0, 0, 0};

//     //We'll generate the final output, 128 bits (16 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (16/sizeof(word_t))) {

//         uint64_t final_bits0 = 0;
//         uint64_t final_bits1 = 0;
//         uint64_t final_mask0 = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed
//         uint64_t final_mask1 = 0xFFFFFFFFFFFFFFFF;

//         uint64_t gt_bits[2];
//         uint64_t chunk_mask[2];
//         generate_and_test_2bit_layer_avx512(&primary_threshold_vec, (__m128i*)gt_bits, (__m128i*)chunk_mask);

//         uint64_t write_mask0 = final_mask0 & chunk_mask[0]; //1 for every bit we will write
//         uint64_t write_mask1 = final_mask1 & chunk_mask[1];
//         final_bits0 = final_bits0 | (write_mask0 & gt_bits[0]);
//         final_bits1 = final_bits1 | (write_mask1 & gt_bits[1]);
//         final_mask0 = final_mask0 & ~write_mask0;
//         final_mask1 = final_mask1 & ~write_mask1;

//         //Mop up any unconverged bits (128 bits at a time)
//         for (int layer = 0; layer < num_layers; layer++) {

//             if (final_mask0 == 0 && final_mask1 == 0) {
//                 break;
//             }

//             //Figure out how many random bits we need and which layer has enough to pull them from
//             int needed_bits0 = _mm_popcnt_u64(final_mask0);
//             int needed_bits1 = _mm_popcnt_u64(final_mask1);

//             // layer_n_passes[layer]++;
//             // tot_needed_bits_per_layer[layer] += needed_bits0;
//             // tot_needed_bits_per_layer[layer] += needed_bits1;

//             //Branchless version of:
//             // if (bucket_cnt[layer] + needed_bits > 64) {
//             //     bucket_cnt[layer] = needed_bits;
//             //     bucket_idx[layer]++;
//             // } else {
//             //     bucket_cnt[layer] += needed_bits;
//             // }
//             int new_cnt0 = bucket_cnt0[layer] + needed_bits0;
//             int new_cnt1 = bucket_cnt1[layer] + needed_bits1;
//             int rollover = new_cnt0 > 64 || new_cnt1 > 64;
//             bucket_cnt0[layer] = (rollover * needed_bits0) + ((!rollover) * new_cnt0);
//             bucket_cnt1[layer] = (rollover * needed_bits1) + ((!rollover) * new_cnt1);
//             bucket_idx[layer] += rollover;

//             //Distribute the random bits according to the mask we need
//             uint64_t current_gt_bits0 = _pdep_u64(layer_gt_bits0[bucket_idx[layer]][layer], final_mask0);
//             uint64_t current_gt_bits1 = _pdep_u64(layer_gt_bits1[bucket_idx[layer]][layer], final_mask1);
//             final_bits0 = final_bits0 | current_gt_bits0;
//             final_bits1 = final_bits1 | current_gt_bits1;
//             uint64_t current_mask_bits0 = _pdep_u64(layer_chunk_mask0[bucket_idx[layer]][layer], final_mask0);
//             uint64_t current_mask_bits1 = _pdep_u64(layer_chunk_mask1[bucket_idx[layer]][layer], final_mask1);
//             final_mask0 = final_mask0 & ~current_mask_bits0;
//             final_mask1 = final_mask1 & ~current_mask_bits1;

//             //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//             layer_chunk_mask0[bucket_idx[layer]][layer] = layer_chunk_mask0[bucket_idx[layer]][layer] >> needed_bits0;
//             layer_chunk_mask1[bucket_idx[layer]][layer] = layer_chunk_mask1[bucket_idx[layer]][layer] >> needed_bits1;
//             layer_gt_bits0[bucket_idx[layer]][layer] = layer_gt_bits0[bucket_idx[layer]][layer] >> needed_bits0;
//             layer_gt_bits1[bucket_idx[layer]][layer] = layer_gt_bits1[bucket_idx[layer]][layer] >> needed_bits1;
//         }

//         *(dst + word_offset) = final_bits0;
//         *(dst + word_offset+1) = final_bits1;
//     }

//     // for (int i=0; i<num_layers; i++) {
//     //     //std::cout << "layer: " << i << ", bucket_idx: " << bucket_idx[i] << "\n";
//     //     std::cout << "layer: " << i << ", passes: " << layer_n_passes[i] << ", total needed: " << tot_needed_bits_per_layer[i] << "\n";
//     // }
// }

//******************************************************************************************
// ID: 2-8Mop-DualScalar
// 2-bit initial, then on-demand mop-up using 8 bits of input, 128-output-bits at a time.
// DESCRIPTION: Start with 2 random input bits tested at a time using AVX-512, then use
//  scalar instructions, doubled up, to do the mop up, with 8-bit thresholds
// STATUS: Works.  Perf ~2.15µs
// COMMENTS: Perf breakdown: 600ns generating and applying the initial 2 bit random.
//  750ns initializing the mop-up bits. We actually initialize 24,576 mop-up bits, vs. 16,384 primary bits
//  750ns performing the mop-up.
//******************************************************************************************

// const __m512i TWO_BIT_MASK = _mm512_set1_epi8(0x03);

// inline void generate_and_test_2bit_layer_avx512(__m512i* layer_threshold_vec, __m128i* gt_bits, __m128i* chunk_mask) {

//     //Get 256 bits of true random, which is 2 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out into two 512-bit AVX vectors, so the lower 2 bits of each 8-bit value
//     // are random, while the upper 6 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 2);
//     __m512i dup_random0;
//     ((__m256i*)(&dup_random0))[0] = rand_source0;
//     ((__m256i*)(&dup_random0))[1] = rand_source1;
//     __m512i true_random0 = _mm512_and_epi64(dup_random0, TWO_BIT_MASK);

//     __m256i rand_source2 = _mm256_srli_epi16(rand_source0, 4);
//     __m256i rand_source3 = _mm256_srli_epi16(rand_source0, 6);
//     __m512i dup_random1;
//     ((__m256i*)(&dup_random1))[0] = rand_source2;
//     ((__m256i*)(&dup_random1))[1] = rand_source3;
//     __m512i true_random1 = _mm512_and_epi64(dup_random1, TWO_BIT_MASK);

//     //We have a definitive answer for every 2-bit random that is above or below the threshold value
//     __m128i eq_bits;
//     ((uint64_t*)gt_bits)[0] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)&eq_bits)[0] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)gt_bits)[1] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random1);
//     ((uint64_t*)&eq_bits)[1] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random1);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// inline void generate_and_test_8bit_layer_avx512(__m512i* layer_threshold_vec, __mmask64* gt_bits, uint64_t* chunk_mask) {

//     //Get 512 bits of true random, which is 8 input bits for each output bit
//     __m512i true_random;
//     ((__m256i*)(&true_random))[0] = avx512_pcg32_random_r(&key512);
//     ((__m256i*)(&true_random))[1] = avx512_pcg32_random_r(&key512);

//     //We have a definitive answer for every 8-bit random that is above or below the threshold value
//     *gt_bits = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random);
//     __mmask64 eq_bits = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by finding a 2-bit threshold from the float, to rule out 75% of the bit patterns.
//     uint8_t primary_threshold = 0;
//     __m512i primary_threshold_vec;
//     float_t x = p;
//     for (int bit=1; bit>=0; bit--) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             primary_threshold += 1 << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }
//     primary_threshold_vec = _mm512_set1_epi8(primary_threshold);

//     //Next, initialize 8-bit chunks for the remaining precision we want from the float threshold
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0xF.
//     const int num_layers = 6;
//     uint8_t layer_thresholds[num_layers];
//     __m512i layer_threshold_vecs[num_layers];
//     for (int i=0; i<num_layers; i++) {
//         layer_thresholds[i] = 0;
//         for (int bit=7; bit>=0; bit--) {
//             if (x<0.5) {
//                 x *= 2.0;
//             } else {
//                 layer_thresholds[i] += 1 << bit;
//                 x -= 0.5;
//                 x *= 2.0;
//             }
//         }
//         layer_threshold_vecs[i] = _mm512_set1_epi8(layer_thresholds[i]);
//     }

//     //Pre-Generate some randoms for the mop-up layers
//     const int mop_bucket_cnt = 24; //8192 bits / 4 (primary screen) / 128 (bits_per_value) * 1.5 (safety margin)
//     uint64_t layer_gt_bits0[mop_bucket_cnt][num_layers];
//     uint64_t layer_gt_bits1[mop_bucket_cnt][num_layers];
//     uint64_t layer_chunk_mask0[mop_bucket_cnt][num_layers];
//     uint64_t layer_chunk_mask1[mop_bucket_cnt][num_layers];
//     uint64_t bucket_cnt0[num_layers];
//     uint64_t bucket_cnt1[num_layers];
//     int bucket_idx[num_layers];
//     int bucket_init_vec_cnt[num_layers] = {48, 1, 1, 1, 1, 1};
//     for (int i=0; i<num_layers; i++) {
//         for (int chunk_idx=0; chunk_idx<bucket_init_vec_cnt[i]; chunk_idx++) {
//             uint64_t gt_bits;
//             uint64_t chunk_mask;
//             generate_and_test_8bit_layer_avx512(&layer_threshold_vecs[i], &gt_bits, &chunk_mask);

//             int vec_idx = chunk_idx/2;
//             if (chunk_idx % 2 == 0) {
//                 layer_gt_bits0[vec_idx][i] = gt_bits & 0xFFFFFFFF;
//                 layer_gt_bits1[vec_idx][i] = gt_bits >> 32;
//                 layer_chunk_mask0[vec_idx][i] = chunk_mask & 0xFFFFFFFF;
//                 layer_chunk_mask1[vec_idx][i] = chunk_mask >> 32;
//             } else {
//                 layer_gt_bits0[vec_idx][i] |= gt_bits << 32;
//                 layer_gt_bits1[vec_idx][i] |= gt_bits & 0xFFFFFFFF00000000;
//                 layer_chunk_mask0[vec_idx][i] |= chunk_mask << 32;
//                 layer_chunk_mask1[vec_idx][i] |= chunk_mask & 0xFFFFFFFF00000000;
//             }
//         }
//         bucket_cnt0[i] = 0;
//         bucket_cnt1[i] = 0;
//         bucket_idx[i] = 0;
//     }

//     //We'll generate the final output, 128 bits (16 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (16/sizeof(word_t))) {

//         uint64_t final_bits0 = 0;
//         uint64_t final_bits1 = 0;
//         uint64_t final_mask0 = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed
//         uint64_t final_mask1 = 0xFFFFFFFFFFFFFFFF;

//         uint64_t gt_bits[2];
//         uint64_t chunk_mask[2];
//         generate_and_test_2bit_layer_avx512(&primary_threshold_vec, (__m128i*)gt_bits, (__m128i*)chunk_mask);

//         uint64_t write_mask0 = final_mask0 & chunk_mask[0]; //1 for every bit we will write
//         uint64_t write_mask1 = final_mask1 & chunk_mask[1];
//         final_bits0 = final_bits0 | (write_mask0 & gt_bits[0]);
//         final_bits1 = final_bits1 | (write_mask1 & gt_bits[1]);
//         final_mask0 = final_mask0 & ~write_mask0;
//         final_mask1 = final_mask1 & ~write_mask1;

//         //Mop up any unconverged bits (128 bits at a time)
//         for (int layer = 0; layer < num_layers; layer++) {

//             if (final_mask0 == 0 && final_mask1 == 0) {
//                 break;
//             }

//             //Figure out how many random bits we need and which layer has enough to pull them from
//             int needed_bits0 = _mm_popcnt_u64(final_mask0);
//             int needed_bits1 = _mm_popcnt_u64(final_mask1);

//             //Branchless version of:
//             // if (bucket_cnt[layer] + needed_bits > 64) {
//             //     bucket_cnt[layer] = needed_bits;
//             //     bucket_idx[layer]++;
//             // } else {
//             //     bucket_cnt[layer] += needed_bits;
//             // }
//             int new_cnt0 = bucket_cnt0[layer] + needed_bits0;
//             int new_cnt1 = bucket_cnt1[layer] + needed_bits1;
//             int rollover = new_cnt0 > 64 || new_cnt1 > 64;
//             bucket_cnt0[layer] = (rollover * needed_bits0) + ((!rollover) * new_cnt0);
//             bucket_cnt1[layer] = (rollover * needed_bits1) + ((!rollover) * new_cnt1);
//             bucket_idx[layer] += rollover;

//             //Distribute the random bits according to the mask we need
//             uint64_t current_gt_bits0 = _pdep_u64(layer_gt_bits0[bucket_idx[layer]][layer], final_mask0);
//             uint64_t current_gt_bits1 = _pdep_u64(layer_gt_bits1[bucket_idx[layer]][layer], final_mask1);
//             final_bits0 = final_bits0 | current_gt_bits0;
//             final_bits1 = final_bits1 | current_gt_bits1;
//             uint64_t current_mask_bits0 = _pdep_u64(layer_chunk_mask0[bucket_idx[layer]][layer], final_mask0);
//             uint64_t current_mask_bits1 = _pdep_u64(layer_chunk_mask1[bucket_idx[layer]][layer], final_mask1);
//             final_mask0 = final_mask0 & ~current_mask_bits0;
//             final_mask1 = final_mask1 & ~current_mask_bits1;

//             //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//             layer_chunk_mask0[bucket_idx[layer]][layer] = layer_chunk_mask0[bucket_idx[layer]][layer] >> needed_bits0;
//             layer_chunk_mask1[bucket_idx[layer]][layer] = layer_chunk_mask1[bucket_idx[layer]][layer] >> needed_bits1;
//             layer_gt_bits0[bucket_idx[layer]][layer] = layer_gt_bits0[bucket_idx[layer]][layer] >> needed_bits0;
//             layer_gt_bits1[bucket_idx[layer]][layer] = layer_gt_bits1[bucket_idx[layer]][layer] >> needed_bits1;
//         }

//         //Write the final bits to the destination
//         *(dst + word_offset) = final_bits0;
//         *(dst + word_offset+1) = final_bits1;
//     }

//     // for (int i=0; i<num_layers; i++) {
//     //     std::cout << "layer: " << i << ", bucket_idx: " << bucket_idx[i] << "\n";
//     //     // std::cout << "layer: " << i << ", passes: " << layer_n_passes[i] << ", total needed: " << tot_needed_bits_per_layer[i] << "\n";
//     // }
// }

//******************************************************************************************
// ID: 2-8Pass-DualScalar
// 2-pass Imprecise.  (2, 8)
// DESCRIPTION: Start with 2 random input bits tested at a time using AVX-512, then use
//  scalar instructions, doubled up, to do one mop up, with 8-bit thresholds
//  This function is only capable of an accuracy up to 1 part in 1024, and the core systemically
//  underestimates.  So, to adjust for that bias, 0.000488 is added to the threshold.
//  (0.000488 = 1/1024/2, ie half of the systematic bias, to center the bias on 0)
// STATUS: Works, with caveats above. Lower quality statistics than other algorithms. Perf ~1.75µs
// COMMENTS: Perf breakdown: 650ns generating and applying the initial 2 bit random.
//  650ns initializing the 2nd pass bits. Same mop-up allocation as 2-8Mop-DualScalar
//  450ns 2nd Pass
//  50ns building map, & other overheads
//******************************************************************************************

// const __m512i TWO_BIT_MASK = _mm512_set1_epi8(0x03);

// inline void generate_and_test_2bit_layer_avx512(__m512i* layer_threshold_vec, __m128i* gt_bits, __m128i* chunk_mask) {

//     //Get 256 bits of true random, which is 2 input bits for each output bit
//     __m256i rand_source0 = avx512_pcg32_random_r(&key512);

//     //Spread the random out into two 512-bit AVX vectors, so the lower 2 bits of each 8-bit value
//     // are random, while the upper 6 bits are 0
//     __m256i rand_source1 = _mm256_srli_epi16(rand_source0, 2);
//     __m512i dup_random0;
//     ((__m256i*)(&dup_random0))[0] = rand_source0;
//     ((__m256i*)(&dup_random0))[1] = rand_source1;
//     __m512i true_random0 = _mm512_and_epi64(dup_random0, TWO_BIT_MASK);

//     __m256i rand_source2 = _mm256_srli_epi16(rand_source0, 4);
//     __m256i rand_source3 = _mm256_srli_epi16(rand_source0, 6);
//     __m512i dup_random1;
//     ((__m256i*)(&dup_random1))[0] = rand_source2;
//     ((__m256i*)(&dup_random1))[1] = rand_source3;
//     __m512i true_random1 = _mm512_and_epi64(dup_random1, TWO_BIT_MASK);

//     //We have a definitive answer for every 2-bit random that is above or below the threshold value
//     __m128i eq_bits;
//     ((uint64_t*)gt_bits)[0] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)&eq_bits)[0] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random0);
//     ((uint64_t*)gt_bits)[1] = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random1);
//     ((uint64_t*)&eq_bits)[1] = _mm512_cmpeq_epu8_mask(*layer_threshold_vec, true_random1);

//     *chunk_mask = *gt_bits | ~eq_bits; //bits that we can definitively answer with this chunk
// }

// inline void generate_and_test_8bit_layer_avx512(__m512i* layer_threshold_vec, __mmask64* gt_bits) {

//     //Get 512 bits of true random, which is 8 input bits for each output bit
//     __m512i true_random;
//     ((__m256i*)(&true_random))[0] = avx512_pcg32_random_r(&key512);
//     ((__m256i*)(&true_random))[1] = avx512_pcg32_random_r(&key512);

//     //We have a definitive answer for every 8-bit random that is above or below the threshold value
//     *gt_bits = _mm512_cmpgt_epu8_mask(*layer_threshold_vec, true_random);
// }

// void random_into_avx512(word_t* dst, float_t p) {

//     //First start by finding a 2-bit threshold from the float, to rule out 75% of the bit patterns.
//     uint8_t primary_threshold = 0;
//     __m512i primary_threshold_vec;
//     float_t x = p + 0.00048828125; //Bias = 1/1024/2;
//     for (int bit=1; bit>=0; bit--) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             primary_threshold += 1 << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }
//     primary_threshold_vec = _mm512_set1_epi8(primary_threshold);

//     //Next, initialize 8-bit chunks for the remaining precision we want from the float threshold
//     //Coming out of this loop, each layer threshold can be any value, 0x0 to 0xF.
//     uint8_t secondary_threshold;
//     __m512i secondary_threshold_vec;
//     secondary_threshold = 0;
//     for (int bit=7; bit>=0; bit--) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             secondary_threshold += 1 << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }
//     secondary_threshold_vec = _mm512_set1_epi8(secondary_threshold);

//     //Pre-Generate some randoms for the secondary pass
//     const int mop_bucket_cnt = 24; //8192 bits / 4 (primary screen) / 128 (bits_per_value) * 1.5 (safety margin)
//     uint64_t layer_gt_bits0[mop_bucket_cnt];
//     uint64_t layer_gt_bits1[mop_bucket_cnt];
//     uint64_t bucket_cnt0;
//     uint64_t bucket_cnt1;
//     int bucket_idx;
//     for (int vec_idx=0; vec_idx<mop_bucket_cnt; vec_idx++) {
//         generate_and_test_8bit_layer_avx512(&secondary_threshold_vec,
//             &layer_gt_bits0[vec_idx]);
//         generate_and_test_8bit_layer_avx512(&secondary_threshold_vec,
//             &layer_gt_bits1[vec_idx]);
//     }
//     bucket_cnt0 = 0;
//     bucket_cnt1 = 0;
//     bucket_idx = 0;

//     //We'll generate the final output, 128 bits (16 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (16/sizeof(word_t))) {

//         uint64_t final_bits0 = 0;
//         uint64_t final_bits1 = 0;
//         uint64_t final_mask0 = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed
//         uint64_t final_mask1 = 0xFFFFFFFFFFFFFFFF;

//         //Primary Pass.  Use 2 input bits for each output.  Only 25% of outputs make it through this filter
//         uint64_t gt_bits[2];
//         uint64_t chunk_mask[2];
//         generate_and_test_2bit_layer_avx512(&primary_threshold_vec, (__m128i*)gt_bits, (__m128i*)chunk_mask);

//         uint64_t write_mask0 = final_mask0 & chunk_mask[0]; //1 for every bit we will write
//         uint64_t write_mask1 = final_mask1 & chunk_mask[1];
//         final_bits0 = final_bits0 | (write_mask0 & gt_bits[0]);
//         final_bits1 = final_bits1 | (write_mask1 & gt_bits[1]);
//         final_mask0 = final_mask0 & ~write_mask0;
//         final_mask1 = final_mask1 & ~write_mask1;

//         //Secondary Pass.  Use 8 input bits for each remaining output.
//         // This filter will provide a diffinitive answer for 99.6% of remaining bits (255/256)
//         // So only 1/1024 will be unanswered by the time we have passed both filters

//         //Figure out how many random bits we need and which layer has enough to pull them from
//         int needed_bits0 = _mm_popcnt_u64(final_mask0);
//         int needed_bits1 = _mm_popcnt_u64(final_mask1);

//         //Branchless version of:
//         // if (bucket_cnt[layer] + needed_bits > 64) {
//         //     bucket_cnt[layer] = needed_bits;
//         //     bucket_idx[layer]++;
//         // } else {
//         //     bucket_cnt[layer] += needed_bits;
//         // }
//         int new_cnt0 = bucket_cnt0 + needed_bits0;
//         int new_cnt1 = bucket_cnt1 + needed_bits1;
//         int rollover = new_cnt0 > 64 || new_cnt1 > 64;
//         bucket_cnt0 = (rollover * needed_bits0) + ((!rollover) * new_cnt0);
//         bucket_cnt1 = (rollover * needed_bits1) + ((!rollover) * new_cnt1);
//         bucket_idx += rollover;

//         //Distribute the random bits according to the mask we need
//         uint64_t current_gt_bits0 = _pdep_u64(layer_gt_bits0[bucket_idx], final_mask0);
//         uint64_t current_gt_bits1 = _pdep_u64(layer_gt_bits1[bucket_idx], final_mask1);
//         final_bits0 = final_bits0 | current_gt_bits0;
//         final_bits1 = final_bits1 | current_gt_bits1;

//         //Shift our mop-up bits, so fresh bits are waiting in the LSBs
//         layer_gt_bits0[bucket_idx] = layer_gt_bits0[bucket_idx] >> needed_bits0;
//         layer_gt_bits1[bucket_idx] = layer_gt_bits1[bucket_idx] >> needed_bits1;

//         *(dst + word_offset) = final_bits0;
//         *(dst + word_offset+1) = final_bits1;
//     }
// }


//******************************************************************************************
// ID: Single-Bit-Parsimonious-Scalar
// DESCRIPTION: Advance single-bit at a time across a 64-bit wavefront.  Only pull needed
//  bits, using a double-buffer for new bits.
// STATUS: Works.  Perf ~4.8µs
// COMMENTS: This is the most bit-frugal algorithm, wasting very few of the input bits,
//  however it loses a lot on management overhead.
//******************************************************************************************

// //single-bit early-out (two reserves)
// void random_into_scalar(word_t* dst, float_t p) {
//     const int bit_precision = 48;

//     //Algorithm
//     //0. Final mask starts out at 0xFFFFFFFFFFFFFFFF
//     //1. Pull 64 random bits
//     //2. All low bits in the random bits should be set either high or low, depending on the layer
//     //3. All low bits should be &(anded) into the final mask
//     //4. Use popcnt on the final mask to figure out how many random bits to pull for the next layer
//     //5. Use pdep with the final mask, to distribute those randoms into the right place for the next cycle

//     //First start by constructing a map of single-bit values, each bit indicating whether
//     // the output value for a low-bit in the input random is high or low.
//     uint64_t map = 0;
//     float_t x = p;
//     for (int bit=0; bit<bit_precision; bit++) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             map |= 0x1LL << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }
//     // std::cout << "p = " << p << ", map = " << std::hex << map <<"\n";

//     //rand_source holds 256 bits of random, and bits from it are accessed 64 at at time
//     __m256i rand_source;
//     int rand_blocks = 0;

//     //reserve_bits0 holds bits we use for operations that don't consume a whole 64-bit block
//     uint64_t reserve_bits0;
//     uint64_t reserve_bits1;
//     int reserve_cnt0 = 0;
//     int reserve_cnt1 = 0;

//     //We'll generate the final output, 64 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (8/sizeof(word_t))) {

//         int bit_idx = 0;
//         uint64_t final_bits = 0;
//         uint64_t final_mask = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed

//         //Get a fresh block of 64 random input bits to start with
//         if (rand_blocks == 0) {
//             rand_source = avx512_pcg32_random_r(&key512);
//             rand_blocks = 3;
//         } else {
//             rand_blocks --;
//         }
//         uint64_t current_mask = ((uint64_t*)&rand_source)[rand_blocks];

//         // reserve_bits0 = ((uint64_t*)&goat_source)[0]; //GOAT, this is crap
//         int needed_bits = 64;

//         //Loop until we've finalized these output 64 bits
//         while (final_mask != 0 && bit_idx < bit_precision) {

//             //Make sure reserve_bits0 is topped up
//             while (reserve_cnt0 < 64) {
//                 if (reserve_cnt1 == 0) {
//                     if (rand_blocks != 0) {
//                         rand_blocks --;
//                     } else {
//                         rand_source = avx512_pcg32_random_r(&key512);
//                         rand_blocks = 3;
//                     }
//                     reserve_bits1 = ((uint64_t*)&rand_source)[rand_blocks];
//                     reserve_cnt1 = 64;
//                 }

//                 int accept_cnt = 64-reserve_cnt0;
//                 if (accept_cnt < reserve_cnt1) {
//                     uint64_t mask = (0xFFFFFFFFFFFFFFFF << reserve_cnt0);
// // std::cout << "GOAT case -A- bits_to_move: " << std::dec << accept_cnt << ", mask: " << std::hex << mask << "\n";
// // std::cout << "GOAT before reserve_bits0: " << std::hex << reserve_bits0 << "\n";

//                     reserve_bits0 = reserve_bits0 | (reserve_bits1 & mask);
//                     reserve_bits1 = reserve_bits1 << (accept_cnt);
// // std::cout << "GOAT after reserve_bits0: " << std::hex << reserve_bits0 << "\n";

//                     reserve_cnt1 -= accept_cnt;
//                     reserve_cnt0 = 64;
//                 } else {
//                     uint64_t mask = (0xFFFFFFFFFFFFFFFF << reserve_cnt0);
// // std::cout << "GOAT CASE (B) bits_to_move: " << std::dec << reserve_cnt1 << ", mask: " << std::hex << mask << "\n";
// // std::cout << "GOAT before reserve_bits0: " << std::hex << reserve_bits0 << "\n";

//                     reserve_bits0 = reserve_bits0 | ((reserve_bits1 >> (64 - reserve_cnt1 - reserve_cnt0))  & mask);

// // std::cout << "GOAT after reserve_bits0: " << std::hex << reserve_bits0 << "\n";

//                     reserve_cnt0 += reserve_cnt1;
//                     reserve_cnt1 = 0;
//                 }
//             }

// // std::cout << "GOAT Starting final mask: " << final_mask << "\n";
// // std::cout << "GOAT bit val: " << ((map >> bit_idx) & 0x1) << " old final_bits " << final_bits << "\n";

//             if ((map >> bit_idx) & 0x1) {
//                 final_bits = final_bits | (~current_mask & final_mask);
//             }
//             final_mask = final_mask & current_mask;

// // std::cout << "GOAT bit idx: " << bit_idx << ", final bits so far: " << final_bits << "\n";

// // std::cout << "GOAT zz OLD current_mask: " << current_mask << "\n";

// // std::cout << "GOAT zz final mask: " << final_mask << ", reserve_bits0: " << reserve_bits0 << "\n";

//             current_mask = _pdep_u64(reserve_bits0, final_mask);

//             needed_bits = _mm_popcnt_u64(final_mask);
//             reserve_bits0 = reserve_bits0 >> needed_bits;
//             reserve_cnt0 -= needed_bits;

// // std::cout << "GOAT reserve_bits0: " << std::hex << reserve_bits0 << ", needed: " << needed_bits << ", count: " << reserve_cnt0 << "\n";

// // std::cout << "GOAT zz new current_mask: " << current_mask << ", new_reserve: " << reserve_bits0 << "\n";


//             bit_idx++;
//         }

// // std::cout << "GOAT Terminated bit: " << bit_idx << " Final bits: " << final_bits << "\n";

//         *((uint64_t*)(dst + word_offset)) = final_bits;
//     }
// }


//******************************************************************************************
// ID: Single-Bit-Single-Buffer-Scalar
// DESCRIPTION: Advance single-bit at a time across a 64-bit wavefront.  Use a single buffer
//  rather than a double-buffer, which wastes more bits but simplifies logic
// STATUS: Works.  Perf ~4.4µs
// COMMENTS: Single-buffering input bits is a win, but not enough.
//******************************************************************************************

// //single-bit early-out, 64-bits at a time (one reserve)
// void random_into_scalar(word_t* dst, float_t p) {
//     const int bit_precision = 48;

//     //First start by constructing a map of single-bit values, each bit indicating whether
//     // the output value for a low-bit in the input random is high or low.
//     uint64_t map = 0;
//     float_t x = p;
//     for (int bit=0; bit<bit_precision; bit++) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             map |= 0x1LL << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }

//     //rand_source holds 256 bits of random, and bits from it are accessed 64 at at time
//     __m256i rand_source;
//     int rand_blocks = 0;

//     //We'll generate the final output, 64 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (8/sizeof(word_t))) {

//         int bit_idx = 0;
//         uint64_t final_bits = 0;
//         uint64_t final_mask = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed

//         //Get a fresh block of 64 random input bits to start with
//         if (rand_blocks == 0) {
//             rand_source = avx512_pcg32_random_r(&key512);
//             rand_blocks = 3;
//         } else {
//             rand_blocks --;
//         }
//         uint64_t current_mask = ((uint64_t*)&rand_source)[rand_blocks];

//         //reserve_bits holds bits we use for operations that don't consume a whole 64-bit block
//         if (rand_blocks == 0) {
//             rand_source = avx512_pcg32_random_r(&key512);
//             rand_blocks = 3;
//         } else {
//             rand_blocks --;
//         }
//         uint64_t reserve_bits = ((uint64_t*)&rand_source)[rand_blocks];
//         int reserve_cnt = 64;

//         //Loop until we've finalized these output 64 bits
//         while (bit_idx < bit_precision) {

//             if ((map >> bit_idx) & 0x1) {
//                 final_bits = final_bits | (~current_mask & final_mask);
//             }
//             final_mask = final_mask & current_mask;

//             if (final_mask == 0) {
//                 break;
//             }

//             current_mask = _pdep_u64(reserve_bits, final_mask);

//             int needed_bits = _mm_popcnt_u64(final_mask);
//             reserve_bits = reserve_bits >> needed_bits;
//             reserve_cnt -= needed_bits;

//             //Make sure reserve_bits is topped up
//             if (reserve_cnt < needed_bits) {
//                 if (rand_blocks != 0) {
//                     rand_blocks --;
//                 } else {
//                     rand_source = avx512_pcg32_random_r(&key512);
//                     rand_blocks = 3;
//                 }
//                 reserve_bits = ((uint64_t*)&rand_source)[rand_blocks];
//                 reserve_cnt = 64;
//             }

//             bit_idx++;
//         }

//         *((uint64_t*)(dst + word_offset)) = final_bits;
//     }
// }

// ******************************************************************************************
// ID: Trash
// STATUS: Works.  4.7µs
// COMMENTS: I can't remember what I was trying with this version.  It's correct but not fast
// ******************************************************************************************

// //single-bit early-out, 64-bits at a time (two reserves)
// void random_into_scalar(word_t* dst, float_t p) {
//     const int bit_precision = 48;

//     //First start by constructing a map of single-bit values, each bit indicating whether
//     // the output value for a low-bit in the input random is high or low.
//     uint64_t map = 0;
//     float_t x = p;
//     for (int bit=0; bit<bit_precision; bit++) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             map |= 0x1LL << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }

//     //rand_source holds 256 bits of random, and bits from it are accessed 64 at at time
//     __m256i rand_source;
//     int rand_blocks = 0;

//     //reserve_bits0 holds bits we use for operations that don't consume a whole 64-bit block
//     rand_source = avx512_pcg32_random_r(&key512);
//     uint64_t reserve_bits0 = ((uint64_t*)&rand_source)[3];
//     uint64_t reserve_bits1 = ((uint64_t*)&rand_source)[2];
//     rand_blocks = 2;
//     int reserve_cnt0 = 64;
//     int reserve_cnt1 = 64;

//     //We'll generate the final output, 64 bits at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (8/sizeof(word_t))) {

//         int bit_idx = 0;
//         uint64_t final_bits = 0;
//         uint64_t final_mask = 0xFFFFFFFFFFFFFFFF; //1 for every bit whose value is still needed

//         //Get a fresh block of 64 random input bits to start with
//         if (rand_blocks == 0) {
//             rand_source = avx512_pcg32_random_r(&key512);
//             rand_blocks = 3;
//         } else {
//             rand_blocks --;
//         }
//         uint64_t current_mask = ((uint64_t*)&rand_source)[rand_blocks];

//         //Loop until we've finalized these output 64 bits
//         while (bit_idx < bit_precision) {

//             if ((map >> bit_idx) & 0x1) {
//                 final_bits = final_bits | (~current_mask & final_mask);
//             }
//             final_mask = final_mask & current_mask;

//             if (final_mask == 0) {
//                 break;
//             }

//             current_mask = _pdep_u64(reserve_bits0, final_mask);

//             int needed_bits = _mm_popcnt_u64(final_mask);
//             reserve_bits0 = reserve_bits0 >> needed_bits;
//             reserve_cnt0 -= needed_bits;

//             //Make sure reserve_bits0 is topped up
//             int accept_cnt = 64-reserve_cnt0;

//             if (reserve_cnt1 < accept_cnt) {
//                 if (rand_blocks != 0) {
//                     rand_blocks --;
//                 } else {
//                     rand_source = avx512_pcg32_random_r(&key512);
//                     rand_blocks = 3;
//                 }
//                 reserve_bits1 = ((uint64_t*)&rand_source)[rand_blocks];
//                 reserve_cnt1 = 64;
//             }

//             uint64_t mask = (0xFFFFFFFFFFFFFFFF << reserve_cnt0);

//             reserve_bits0 = reserve_bits0 | (reserve_bits1 & mask);
//             reserve_bits1 = reserve_bits1 << (accept_cnt);

//             reserve_cnt1 -= accept_cnt;
//             reserve_cnt0 = 64;

//             bit_idx++;
//         }

//         *((uint64_t*)(dst + word_offset)) = final_bits;
//     }
// }


//******************************************************************************************
// ID: Single-Bit-Single-Buffer-AVX2
// DESCRIPTION: Advance single-bit at a time across a 256-bit wavefront.  Use a single buffer
//  rather than a double-buffer, which wastes more bits but simplifies logic
// STATUS: Works.  Perf ~2.7µs
// COMMENTS:  This algorithm ends up throwing away ~33% of all input bits because the reserve
//  buffer is slightly over-demanded, so we chuck it, but then the backup is only half-consumed
//  because we reset it each chunk.  So on average, we chuck a whole buffer each chunk,
//  bringing the input bit consumpion up to 3 inputs per output, vs. the theoretical 2.
//  Ie. we waste 1/3 of the total bits.
//  This perf is decent, considering.  It's possible that combining this AVX2 code path
//  with the double-buffer appraoch in Single-Bit-Parsimonious-Scalar could yield a new
//  winner, because the logic cost is reduced by 4, while the bit savings is constant.
//******************************************************************************************

// //DESCRIPTION: single-bit early-out, 256 bits at a time
// //STATUS: working.  Fast, but not screaming fast.  
// void random_into_avx2(word_t* dst, float_t p) {
//     const int bit_precision = 48;

//     const __m256i zero_vec = _mm256_set1_epi64x(0);
//     const __m256i full_vec = _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF);
//     const __m256i sixty_four_vec = _mm256_set1_epi64x(64);

//     //First start by constructing a map of single-bit values, each bit indicating whether
//     // the output value for a low-bit in the input random is high or low.
//     uint64_t map = 0;
//     float_t x = p;
//     for (int bit=0; bit<bit_precision; bit++) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             map |= 0x1LL << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }

//     //We'll generate the final output, 256 bits (32 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (32/sizeof(word_t))) {

//         int bit_idx = 0;
//         __m256i final_bits = zero_vec;
//         __m256i final_mask = full_vec; //1 for every bit whose value is still needed

//         //Get a fresh block of random input bits to start with
//         // aes_state += increment;
//         // __m256i intermediate_rand = _mm256_aesenc_epi128(aes_state, increment);
//         // __m256i current_mask = _mm256_aesenc_epi128(intermediate_rand, increment);
//         __m256i current_mask = avx2_pcg32_random_r(&avx2_key);

//         //reserve_bits holds bits we use for operations that don't consume a whole 64-bit block
//         // aes_state += increment;
//         // intermediate_rand = _mm256_aesenc_epi128(aes_state, increment);
//         // __m256i reserve_bits = _mm256_aesenc_epi128(intermediate_rand, increment);
//         __m256i reserve_bits = avx2_pcg32_random_r(&avx2_key);
//         __m256i reserve_cnt = sixty_four_vec;

//         //Loop until we've finalized these 256 output bits
//         while (bit_idx < bit_precision) {

//             if ((map >> bit_idx) & 0x1) {
//                 //final_bits = final_bits | (~current_mask & final_mask);
//                 __m256i write_mask = _mm256_andnot_si256(current_mask, final_mask);
//                 final_bits = _mm256_or_si256(final_bits, write_mask);
//             }
//             //final_mask = final_mask & current_mask;
//             final_mask = _mm256_and_si256(final_mask, current_mask);

//             //if (final_mask == 0) {
//             if (_mm256_testz_si256(final_mask, final_mask)) {
//                 break;
//             }

//             //GOAT NOTE: I wish we had a vector PDEP!!  Apparently ARM does.
//             ((uint64_t*)(&current_mask))[0] = _pdep_u64(((uint64_t*)(&reserve_bits))[0], ((uint64_t*)(&final_mask))[0]);
//             ((uint64_t*)(&current_mask))[1] = _pdep_u64(((uint64_t*)(&reserve_bits))[1], ((uint64_t*)(&final_mask))[1]);
//             ((uint64_t*)(&current_mask))[2] = _pdep_u64(((uint64_t*)(&reserve_bits))[2], ((uint64_t*)(&final_mask))[2]);
//             ((uint64_t*)(&current_mask))[3] = _pdep_u64(((uint64_t*)(&reserve_bits))[3], ((uint64_t*)(&final_mask))[3]);

//             //int needed_bits = _mm_popcnt_u64(final_mask);
//             //GOAT NOTE: Looks like I need a "Saphire Rapids" chip to use _mm256_popcnt_epi64, and similar instructions
//             //UPDATE: Apparently Adam can use them on IceLake.
//             __m256i needed_bits;
//             ((uint64_t*)(&needed_bits))[0] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[0]);
//             ((uint64_t*)(&needed_bits))[1] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[1]);
//             ((uint64_t*)(&needed_bits))[2] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[2]);
//             ((uint64_t*)(&needed_bits))[3] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[3]);

//             //reserve_bits = reserve_bits >> needed_bits;
//             reserve_bits = _mm256_srlv_epi64(reserve_bits, needed_bits);

//             //reserve_cnt -= needed_bits;
//             reserve_cnt = _mm256_sub_epi64(reserve_cnt, needed_bits);

//             //Make sure reserve_bits0 is topped up
//             // if (reserve_cnt < needed_bits) {
//             __m256i cmp_results = _mm256_cmpgt_epi8(needed_bits, reserve_cnt);
//             if (_mm256_testnzc_si256(cmp_results, full_vec)) {

//                 // aes_state += increment;
//                 // intermediate_rand = _mm256_aesenc_epi128(aes_state, increment);
//                 // reserve_bits = _mm256_aesenc_epi128(intermediate_rand, increment);
//                 reserve_bits = avx2_pcg32_random_r(&avx2_key);
//                 reserve_cnt = sixty_four_vec;
//             }

//             bit_idx++;
//         }

//         *((__m256i*)(dst + word_offset)) = final_bits;
//     }
// }

//******************************************************************************************
// ID: Single-Bit-Single-Buffer-AVX-512
// DESCRIPTION: Advance single-bit at a time across a 512-bit wavefront.  Use a single buffer
//  rather than a double-buffer, which wastes more bits but simplifies logic
// STATUS: Works.  Perf ~2.8µs
// COMMENTS:  This version is (slightly) SLOWER! than the AVX2 counterpart.  Presumably
//  because working with 512-bits at a time doesn't give the processor as much opportunity
//  to hide pdep and popcnt hotspots.
//******************************************************************************************

// //DESCRIPTION: single-bit early-out, 512 bits at a time
// void random_into_avx512(word_t* dst, float_t p) {
//     const int bit_precision = 48;

//     const __m512i zero_vec = _mm512_set1_epi64(0);
//     const __m512i full_vec = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
//     const __m512i sixty_four_vec = _mm512_set1_epi64(64);

//     //First start by constructing a map of single-bit values, each bit indicating whether
//     // the output value for a low-bit in the input random is high or low.
//     uint64_t map = 0;
//     float_t x = p;
//     for (int bit=0; bit<bit_precision; bit++) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             map |= 0x1LL << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }

//     //We'll generate the final output, 512 bits (64 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (64/sizeof(word_t))) {

//         int bit_idx = 0;
//         __m512i final_bits = zero_vec;
//         __m512i final_mask = full_vec; //1 for every bit whose value is still needed

//         //Get a fresh block of random input bits to start with
//         __m512i current_mask;
//         ((__m256i*)(&current_mask))[0] = avx512_pcg32_random_r(&key512);
//         ((__m256i*)(&current_mask))[1] = avx512_pcg32_random_r(&key512);

//         //reserve_bits holds bits we use for operations that don't consume a whole 64-bit block
//         __m512i reserve_bits;
//         ((__m256i*)(&reserve_bits))[0] = avx512_pcg32_random_r(&key512);
//         ((__m256i*)(&reserve_bits))[1] = avx512_pcg32_random_r(&key512);
//         __m512i reserve_cnt = sixty_four_vec;

//         //Loop until we've finalized these 256 output bits
//         while (bit_idx < bit_precision) {

//             if ((map >> bit_idx) & 0x1) {
//                 //final_bits = final_bits | (~current_mask & final_mask);
//                 __m512i write_mask = _mm512_andnot_si512(current_mask, final_mask);
//                 final_bits = _mm512_or_si512(final_bits, write_mask);
//             }
//             //final_mask = final_mask & current_mask;
//             final_mask = _mm512_and_si512(final_mask, current_mask);

//             //if (final_mask == 0) {
//             //GOAT, there's gotta be a nicer / cheaper way than this!!!
//             if (_mm256_testz_si256(((__m256i*)(&final_mask))[0], ((__m256i*)(&final_mask))[0]) &&
//                 _mm256_testz_si256(((__m256i*)(&final_mask))[1], ((__m256i*)(&final_mask))[1])) {
//                 break;
//             }

//             //GOAT NOTE: I wish we had a vector PDEP!!  Apparently ARM does.
//             ((uint64_t*)(&current_mask))[0] = _pdep_u64(((uint64_t*)(&reserve_bits))[0], ((uint64_t*)(&final_mask))[0]);
//             ((uint64_t*)(&current_mask))[1] = _pdep_u64(((uint64_t*)(&reserve_bits))[1], ((uint64_t*)(&final_mask))[1]);
//             ((uint64_t*)(&current_mask))[2] = _pdep_u64(((uint64_t*)(&reserve_bits))[2], ((uint64_t*)(&final_mask))[2]);
//             ((uint64_t*)(&current_mask))[3] = _pdep_u64(((uint64_t*)(&reserve_bits))[3], ((uint64_t*)(&final_mask))[3]);
//             ((uint64_t*)(&current_mask))[4] = _pdep_u64(((uint64_t*)(&reserve_bits))[4], ((uint64_t*)(&final_mask))[4]);
//             ((uint64_t*)(&current_mask))[5] = _pdep_u64(((uint64_t*)(&reserve_bits))[5], ((uint64_t*)(&final_mask))[5]);
//             ((uint64_t*)(&current_mask))[6] = _pdep_u64(((uint64_t*)(&reserve_bits))[6], ((uint64_t*)(&final_mask))[6]);
//             ((uint64_t*)(&current_mask))[7] = _pdep_u64(((uint64_t*)(&reserve_bits))[7], ((uint64_t*)(&final_mask))[7]);

//             //int needed_bits = _mm_popcnt_u64(final_mask);
//             __m512i needed_bits = _mm512_popcnt_epi64(final_mask);

//             //reserve_bits = reserve_bits >> needed_bits;
//             reserve_bits = _mm512_srlv_epi64(reserve_bits, needed_bits);

//             //reserve_cnt -= needed_bits;
//             reserve_cnt = _mm512_sub_epi64(reserve_cnt, needed_bits);

//             //Make sure reserve_bits0 is topped up
//             // if (reserve_cnt < needed_bits) {
//             __mmask64 cmp_results = _mm512_cmpgt_epu8_mask(needed_bits, reserve_cnt);
//             if (cmp_results != 0) {
//                 ((__m256i*)(&reserve_bits))[0] = avx512_pcg32_random_r(&key512);
//                 ((__m256i*)(&reserve_bits))[1] = avx512_pcg32_random_r(&key512);
//                 reserve_cnt = sixty_four_vec;
//             }

//             bit_idx++;
//         }

//         *((__m512i*)(dst + word_offset)) = final_bits;
//     }
// }

//******************************************************************************************
// ID: Single-Bit-Parsimonious-AVX2
// DESCRIPTION: Attempt to unify Single-Bit-Single-Buffer-AVX2 and Single-Bit-Parsimonious-Scalar
// STATUS: NOT WORKING
// COMMENTS:  
//******************************************************************************************

// void random_into_avx2(word_t* dst, float_t p) {
//     const int bit_precision = 48;

//     //First start by constructing a map of single-bit values, each bit indicating whether
//     // the output value for a low-bit in the input random is high or low.
//     uint64_t map = 0;
//     float_t x = p;
//     for (int bit=0; bit<bit_precision; bit++) {
//         if (x<0.5) {
//             x *= 2.0;
//         } else {
//             map |= 0x1LL << bit;
//             x -= 0.5;
//             x *= 2.0;
//         }
//     }

//     //We'll generate the final output, 256 bits (32 Bytes) at a time
//     for (size_t word_offset = 0; word_offset < (BYTES/sizeof(word_t)); word_offset += (32/sizeof(word_t))) {

//         int bit_idx = 0;
//         __m256i final_bits = zero_vec;
//         __m256i final_mask = full_vec; //1 for every bit whose value is still needed

//         //Get a fresh block of random input bits to start with
//         // aes_state += increment;
//         // __m256i intermediate_rand = _mm256_aesenc_epi128(aes_state, increment);
//         // __m256i current_mask = _mm256_aesenc_epi128(intermediate_rand, increment);
//         __m256i current_mask = avx2_pcg32_random_r(&avx2_key);

//         //reserve_bits holds bits we use for operations that don't consume a whole 64-bit block
//         // aes_state += increment;
//         // intermediate_rand = _mm256_aesenc_epi128(aes_state, increment);
//         // __m256i reserve_bits = _mm256_aesenc_epi128(intermediate_rand, increment);
//         __m256i reserve_bits0 = avx2_pcg32_random_r(&avx2_key);
//         __m256i reserve_cnt0 = sixty_four_vec;

//         //Loop until we've finalized these 256 output bits
//         while (bit_idx < bit_precision) {

//             if ((map >> bit_idx) & 0x1) {
//                 //final_bits = final_bits | (~current_mask & final_mask);
//                 __m256i write_mask = _mm256_andnot_si256(current_mask, final_mask);
//                 final_bits = _mm256_or_si256(final_bits, write_mask);
//             }
//             //final_mask = final_mask & current_mask;
//             final_mask = _mm256_and_si256(final_mask, current_mask);

//             //if (final_mask == 0) {
//             if (_mm256_testz_si256(final_mask, final_mask)) {
//                 break;
//             }

//             //GOAT NOTE: I wish we had a vector PDEP!!  Apparently ARM does.
//             ((uint64_t*)(&current_mask))[0] = _pdep_u64(((uint64_t*)(&reserve_bits0))[0], ((uint64_t*)(&final_mask))[0]);
//             ((uint64_t*)(&current_mask))[1] = _pdep_u64(((uint64_t*)(&reserve_bits0))[1], ((uint64_t*)(&final_mask))[1]);
//             ((uint64_t*)(&current_mask))[2] = _pdep_u64(((uint64_t*)(&reserve_bits0))[2], ((uint64_t*)(&final_mask))[2]);
//             ((uint64_t*)(&current_mask))[3] = _pdep_u64(((uint64_t*)(&reserve_bits0))[3], ((uint64_t*)(&final_mask))[3]);

//             //int needed_bits = _mm_popcnt_u64(final_mask);
//             //GOAT NOTE: Looks like I need a "Saphire Rapids" chip to use _mm256_popcnt_epi64, and similar instructions
//             __m256i needed_bits;
//             ((uint64_t*)(&needed_bits))[0] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[0]);
//             ((uint64_t*)(&needed_bits))[1] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[1]);
//             ((uint64_t*)(&needed_bits))[2] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[2]);
//             ((uint64_t*)(&needed_bits))[3] = _mm_popcnt_u64(((uint64_t*)(&final_mask))[3]);

//             //reserve_bits = reserve_bits >> needed_bits;
//             reserve_bits0 = _mm256_srlv_epi64(reserve_bits0, needed_bits);

//             //reserve_cnt -= needed_bits;
//             reserve_cnt0 = _mm256_sub_epi64(reserve_cnt0, needed_bits);

//             //Make sure reserve_bits0 is topped up
//             // if (reserve_cnt < needed_bits) {
//             __m256i cmp_results = _mm256_cmpgt_epi8(needed_bits, reserve_cnt0);
//             if (_mm256_testnzc_si256(cmp_results, full_vec)) {

//                 // aes_state += increment;
//                 // intermediate_rand = _mm256_aesenc_epi128(aes_state, increment);
//                 // reserve_bits = _mm256_aesenc_epi128(intermediate_rand, increment);
//                 reserve_bits = avx2_pcg32_random_r(&avx2_key);
//                 reserve_cnt = sixty_four_vec;
//             }

//             bit_idx++;
//         }

//         *((__m256i*)(dst + word_offset)) = final_bits;
//     }
// }


