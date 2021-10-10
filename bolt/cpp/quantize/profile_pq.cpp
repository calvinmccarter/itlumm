
#include "catch.hpp"
#include "product_quantize.hpp"

#include <string>

#include "eigen_utils.hpp"
#include "timing_utils.hpp"
#include "testing_utils.hpp"
#include "memory.hpp"

#include "debug_utils.hpp"

#define PROFILE_ENCODE
//#define PROFILE_SCAN
//#define PROFILE_QUERY
//#define PROFILE_NONFLOATS

static constexpr int kNreps = 10;
// static constexpr int kNreps = 1;
static constexpr int kNtrials = 5;

static constexpr int M = 32; // number of bytes per compressed vect
static constexpr int64_t nrows_enc = 10*1000; // number of rows to encode
static constexpr int64_t nrows_lut = 10*1000; // number of luts to create
static constexpr int64_t nrows_scan = 1000*1000;
static constexpr int64_t nrows_query = 100*1000;
static constexpr int nqueries = 100;

static constexpr int bits_per_codebook = 8;

static constexpr int ncodebooks = M * (8 / bits_per_codebook);
static constexpr int ncentroids = (1 << bits_per_codebook);
static constexpr int ncentroids_total = ncentroids * ncodebooks;
static constexpr int lut_data_sz = ncentroids * ncodebooks;
static constexpr int subvect_len = 1024 / ncodebooks; // ncols * subvect_len = number of features
//static constexpr int subvect_len = 512 / ncodebooks; // ncols * subvect_len = number of features
//static constexpr int subvect_len = 256 / ncodebooks; // ncols * subvect_len = number of features
// static constexpr int subvect_len = 128 / ncodebooks; // ncols * subvect_len = number of features
//static constexpr int subvect_len = 64 / ncodebooks; // ncols * subvect_len = number of features
static constexpr int ncols = ncodebooks * subvect_len;


TEST_CASE("print pq params", "[pq][mcq][profile]") {
    printf("mcq_D=%d_M=%d\n", ncols, M);  // tmp hack so it suggests the right filename

    printf("------------------------ pq\n");
    printf("---- pq profiling parameters\n");
    printf("pq M: %d\n", M);
    printf("pq nrows_enc: %g\n", (double)nrows_enc);
    printf("pq nrows_lut: %g\n", (double)nrows_lut);
    printf("pq nrows_scan: %g\n", (double)nrows_scan);
    printf("pq nrows_query: %g\n", (double)nrows_query);
    printf("pq subvect_len: %d\n", subvect_len);
    printf("pq ncols: %d\n", ncols);
    printf("pq nqueries: %d\n", nqueries);
    printf("---- pq timings\n");
}

#ifdef PROFILE_ENCODE
TEST_CASE("pq encoding speed", "[pq][mcq][profile]") {
    static constexpr int nrows = nrows_enc;
    // static constexpr int codes_sz = nrows * M;
    // // static constexpr int M = 32;
    // static constexpr int subvect_len = 4;
    // static constexpr int ncodebooks = M;
    // static constexpr int ncentroids = 256;
    // static constexpr int ncentroids_total = ncentroids * ncodebooks;
    // static constexpr int ncols = ncodebooks * subvect_len;
    // static constexpr int lut_data_sz = ncentroids * ncodebooks;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> X(nrows, ncols);
    X.setRandom();
    ColMatrix<uint8_t> codes_out(nrows, M);

    REQUIRE(X.data() != nullptr);
    REQUIRE(X.row(nrows-1).data() != nullptr);

    // PROFILE_DIST_COMPUTATION("pq encode", 5, codes_out.data(), nrows,
    //     pq_encode_8b<M>(X.data(), nrows, ncols, centroids.data(),
    //         codes_out.data()) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "pq encode", kNtrials,
        codes_out.data(), nrows,
        pq_encode_8b<M>(X.data(), nrows, ncols, centroids.data(),
            codes_out.data()) );

    // optimized product quantization (OPQ)
    ColMatrix<float> R(ncols, ncols);
    R.setRandom();
    RowMatrix<float> X_tmp(nrows, ncols);
    // PROFILE_DIST_COMPUTATION("opq encode", 5, codes_out.data(), nrows,
    //     opq_encode_8b<M>(X, centroids.data(), R, X_tmp, codes_out.data()) );
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "opq encode", kNtrials,
        codes_out.data(), nrows,
        opq_encode_8b<M>(X, centroids.data(), R, X_tmp, codes_out.data()) );

    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "pairq encode", kNtrials,
        codes_out.data(), nrows,
        opq_encode_8b<M>(X, centroids.data(), R, X_tmp, codes_out.data()) );
}


TEST_CASE("pq lut encoding speed", "[pq][mcq][profile]") {
     static constexpr int nrows = nrows_lut;
// //   static constexpr int nrows = 200;
//     // static constexpr int M = 32;
//     static constexpr int ncodebooks = M;
//     static constexpr int ncentroids = 256;
//     static constexpr int ncentroids_total = ncentroids * ncodebooks;
//     static constexpr int subvect_len = 4;
//     static constexpr int ncols = ncodebooks * subvect_len;
//     static constexpr int lut_data_sz = ncentroids * ncodebooks;

    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();
    RowMatrix<float> Q(nrows, ncols);

    ColMatrix<uint8_t> lut_out_u8(ncentroids, ncodebooks);
    ColMatrix<uint16_t> lut_out_u16(ncentroids, ncodebooks);
    ColMatrix<float> lut_out_f(ncentroids, ncodebooks);

    // PROFILE_DIST_COMPUTATION_LOOP("pq encode lut 8b dist", 5, lut_out_u8.data(),
    //     lut_data_sz, nrows,
    //     pq_lut_8b<M>(Q.row(i).data(), ncols, centroids.data(), lut_out_u8.data()));
    // PROFILE_DIST_COMPUTATION_LOOP("pq encode lut 16b dist", 5, lut_out_u16.data(),
    //     lut_data_sz, nrows,
    //     pq_lut_8b<M>(Q.row(i).data(), ncols, centroids.data(), lut_out_u16.data()));
    // PROFILE_DIST_COMPUTATION_LOOP("pq encode lut float dist", 5, lut_out_f.data(),
    //     lut_data_sz, nrows,
    //     pq_lut_8b<M>(Q.row(i).data(), ncols, centroids.data(), lut_out_f.data()));

#ifdef PROFILE_NONFLOATS
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pq encode lut 8b dist", kNtrials,
        lut_out_u8.data(), lut_data_sz, nrows,
        pq_lut_8b<M>(Q.row(i).data(), ncols, centroids.data(), lut_out_u8.data()));
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pq encode lut 16b dist", kNtrials,
        lut_out_u16.data(), lut_data_sz, nrows,
        pq_lut_8b<M>(Q.row(i).data(), ncols, centroids.data(), lut_out_u16.data()));
#endif
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pq encode lut float dist", kNtrials,
        lut_out_f.data(), lut_data_sz, nrows,
        pq_lut_8b<M>(Q.row(i).data(), ncols, centroids.data(), lut_out_f.data()));

    // optimized product quantization (OPQ)
    ColMatrix<float> R(ncols, ncols);
    R.setRandom();
    RowVector<float> q_tmp(ncols);

    // PROFILE_DIST_COMPUTATION_LOOP("opq encode lut 8b dist", 5, lut_out_u8.data(),
    //     lut_data_sz, nrows,
    //     opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_u8.data()));
    // PROFILE_DIST_COMPUTATION_LOOP("opq encode lut 16b dist", 5, lut_out_u16.data(),
    //     lut_data_sz, nrows,
    //     opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_u16.data()));
    // PROFILE_DIST_COMPUTATION_LOOP("opq encode lut float dist", 5, lut_out_f.data(),
    //     lut_data_sz, nrows,
    //     opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_f.data()));

#ifdef PROFILE_NONFLOATS
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "opq encode lut 8b dist", kNtrials,
        lut_out_u8.data(), lut_data_sz, nrows,
        opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_u8.data()));
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "opq encode lut 16b dist", kNtrials,
        lut_out_u16.data(), lut_data_sz, nrows,
        opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_u16.data()));
#endif
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "opq encode lut float dist", kNtrials,
        lut_out_f.data(), lut_data_sz, nrows,
        opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_f.data()));

#ifdef PROFILE_NONFLOATS
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pairq encode lut 8b dist", kNtrials,
        lut_out_u8.data(), lut_data_sz, nrows,
        opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_u8.data()));
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pairq encode lut 16b dist", kNtrials,
        lut_out_u16.data(), lut_data_sz, nrows,
        opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_u16.data()));
#endif
    REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pairq encode lut float dist", kNtrials,
        lut_out_f.data(), lut_data_sz, nrows,
        opq_lut_8b<M>(Q.row(i), centroids.data(), R, q_tmp, lut_out_f.data()));


//    PROFILE_DIST_COMPUTATION("opq encode", 5, codes_out.data(), nrows,
//                             opq_encode_8b<M>(X, centroids.data(), R, X_tmp, codes_out.data()) );
}
#endif

#ifdef PROFILE_SCAN
TEST_CASE("pq scan speed", "[pq][mcq][profile]") {
    static constexpr int nrows = nrows_scan;

    // create random codes from in [0, 256]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();

    // create random luts
    ColMatrix<uint8_t> luts_u8(ncentroids, ncodebooks);
    luts_u8.setRandom();
    luts_u8 = luts_u8.array() / (2 * M); // make max lut value small

    ColMatrix<uint16_t> luts_u16(ncentroids, ncodebooks);
    luts_u16.setRandom();
    luts_u16 = luts_u16.array() / (2 * M); // make max lut value small

    ColMatrix<float> luts_f(ncentroids, ncodebooks);
    luts_f.setRandom();
    luts_f = luts_f.array() / (2 * M); // make max lut value small

    // create arrays in which to store the distances
    RowVector<uint8_t> dists_u8(nrows);
    RowVector<uint16_t> dists_u16(nrows);
    RowVector<float> dists_f(nrows);

    // do the scans to compute the distances
    // PROFILE_DIST_COMPUTATION("pq scan uint8", 5, dists_u8.data(), nrows,
    //     pq_scan_8b<M>(codes.data(), luts_u8.data(), dists_u8.data(), nrows));
    // PROFILE_DIST_COMPUTATION("pq scan uint16", 5, dists_u16.data(), nrows,
    //     pq_scan_8b<M>(codes.data(), luts_u16.data(), dists_u16.data(), nrows));
    // PROFILE_DIST_COMPUTATION("pq scan float", 5, dists_f.data(), nrows,
    //     pq_scan_8b<M>(codes.data(), luts_f.data(), dists_f.data(), nrows));

#ifdef PROFILE_NONFLOATS
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "pq scan uint8", kNtrials,
        dists_u8.data(), nrows,
        pq_scan_8b<M>(codes.data(), luts_u8.data(), dists_u8.data(), nrows));
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "pq scan uint16", kNtrials,
        dists_u16.data(), nrows,
        pq_scan_8b<M>(codes.data(), luts_u16.data(), dists_u16.data(), nrows));
#endif
    REPEATED_PROFILE_DIST_COMPUTATION(kNreps, "pq scan float", kNtrials,
        dists_f.data(), nrows,
        pq_scan_8b<M>(codes.data(), luts_f.data(), dists_f.data(), nrows));
}
#endif

#ifdef PROFILE_QUERY
template<int M, class dist_t>
void _run_query(const uint8_t* codes, int nrows,
    const float* q, int ncols,
    const float* centroids,
    dist_t* lut_out, dist_t* dists_out)
{
    pq_lut_8b<M>(q, ncols, centroids, lut_out);
    pq_scan_8b<M>(codes, lut_out, dists_out, nrows);
}

template<int M, class MatrixT, class dist_t>
void _run_query_opq(const uint8_t* codes, int nrows,
                RowVector<float> q,
                const float* centroids,
                const MatrixT& R,
                RowVector<float> q_out,
                dist_t* lut_out, dist_t* dists_out)
{
    opq_lut_8b<M>(q, centroids, R, q_out, lut_out);
    pq_scan_8b<M>(codes, lut_out, dists_out, nrows);
}

TEST_CASE("pq query (lut + scan) speed", "[pq][mcq][profile]") {
    static constexpr int nrows = nrows_query;

    // create random codes from in [0, 256]
    ColMatrix<uint8_t> codes(nrows, ncodebooks);
    codes.setRandom();

    // create random queries
    RowMatrix<float> Q(nqueries, ncols);
    Q.setRandom();

    // create random centroids
    ColMatrix<float> centroids(ncentroids, ncols);
    centroids.setRandom();

    // create random opq rotation
    ColMatrix<float> R(ncols, ncols);
    R.setRandom();
    RowVector<float> q_tmp(ncols);

#ifdef PROFILE_NONFLOATS
    SECTION("uint8_t") {
        ColMatrix<uint8_t> luts(ncentroids, ncodebooks);
        RowVector<uint8_t> dists(nrows);

        // PROFILE_DIST_COMPUTATION_LOOP("pq query u8", 5, dists.data(),
        //     nrows, nqueries,
        //     _run_query<M>(codes.data(), nrows, Q.row(i).data(), ncols,
        //         centroids.data(), luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pq query u8", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query<M>(codes.data(), nrows, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) );

        // PROFILE_DIST_COMPUTATION_LOOP("opq query u8", 5, dists.data(),
        //     nrows, nqueries,
        //     _run_query_opq<M>(codes.data(), nrows, Q.row(i),
        //         centroids.data(), R, q_tmp, luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "opq query u8", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query_opq<M>(codes.data(), nrows, Q.row(i),
                centroids.data(), R, q_tmp, luts.data(), dists.data()) );

        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pairq query u8", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query_opq<M>(codes.data(), nrows, Q.row(i),
                centroids.data(), R, q_tmp, luts.data(), dists.data()) );
    }
    SECTION("uint16_t") {
        ColMatrix<uint16_t> luts(ncentroids, ncodebooks);
        RowVector<uint16_t> dists(nrows);
        // PROFILE_DIST_COMPUTATION_LOOP("pq query u16", 5, dists.data(),
        //     nrows, nqueries,
        //     _run_query<M>(codes.data(), nrows, Q.row(i).data(), ncols,
        //         centroids.data(), luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pq query u16", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query<M>(codes.data(), nrows, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) );

        // PROFILE_DIST_COMPUTATION_LOOP("opq query u16", 5, dists.data(),
        //     nrows, nqueries,
        //     _run_query_opq<M>(codes.data(), nrows, Q.row(i),
        //         centroids.data(), R, q_tmp, luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "opq query u16", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query_opq<M>(codes.data(), nrows, Q.row(i),
                centroids.data(), R, q_tmp, luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pairq query u16", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query_opq<M>(codes.data(), nrows, Q.row(i),
                centroids.data(), R, q_tmp, luts.data(), dists.data()) );
    }
#endif
    SECTION("float") {
        ColMatrix<float> luts(ncentroids, ncodebooks);
        RowVector<float> dists(nrows);
        // PROFILE_DIST_COMPUTATION_LOOP("pq query float", 5, dists.data(),
        //     nrows, nqueries,
        //     (_run_query<M>(codes.data(), nrows, Q.row(i).data(), ncols,
        //         centroids.data(), luts.data(), dists.data()) ));
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pq query float", kNtrials,
            dists.data(), nrows, nqueries,
            (_run_query<M>(codes.data(), nrows, Q.row(i).data(), ncols,
                centroids.data(), luts.data(), dists.data()) ));

        // PROFILE_DIST_COMPUTATION_LOOP("opq query float", 5, dists.data(),
        //     nrows, nqueries,
        //     _run_query_opq<M>(codes.data(), nrows, Q.row(i),
        //         centroids.data(), R, q_tmp, luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "opq query float", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query_opq<M>(codes.data(), nrows, Q.row(i),
                centroids.data(), R, q_tmp, luts.data(), dists.data()) );
        REPEATED_PROFILE_DIST_COMPUTATION_LOOP(kNreps, "pairq query float", kNtrials,
            dists.data(), nrows, nqueries,
            _run_query_opq<M>(codes.data(), nrows, Q.row(i),
                centroids.data(), R, q_tmp, luts.data(), dists.data()) );
    }
}
#endif //PROFILE_QUERY


