#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <cstring>
#include <limits>
#include <cuda_runtime.h>
#include <algorithm>
#include <omp.h>

// Minizip headers
#include "mz.h"
#include "mz_strm.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"

// ---------------------- Type aliases ----------------------
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

// ---------------------- Structs (Must match kernel) ---------------------------
struct pkzip_hash
{
    u8  data_type_enum;
    u8  magic_type_enum;
    u32 compressed_length;
    u32 uncompressed_length;
    u32 crc32;
    u32 offset;
    u32 additional_offset;
    u8  compression_type;
    u32 data_length;
    u16 checksum_from_crc;
    u16 checksum_from_timestamp;
    u32 data[10]; 
};

typedef struct pkzip_hash pkzip_hash_t;

struct pkzip
{
    u8 hash_count;
    u8 checksum_size;
    u8 version;
    pkzip_hash_t hash;
};

typedef struct pkzip pkzip_t;

struct digest_t
{
    u32 digest_buf[4];
};

// External kernel declaration
__global__ void m17200_sxx_cuda_optimized(
    const pkzip_t *esalt_bufs,
    const digest_t *digests_buf,
    const u8 *pw_storage,
    const u32 *pw_offsets,
    u32 gid_cnt,
    u32 *match_out);

// ---------------------- Helpers ---------------------------

u32 hex2int(const std::string& hex) {
    u32 x;
    std::stringstream ss;
    ss << std::hex << hex;
    ss >> x;
    return x;
}

void hex2bin(const std::string& hex, u8* buf, size_t max_len) {
    size_t len = hex.length() / 2;
    if (len > max_len) len = max_len;
    for (size_t i = 0; i < len; i++) {
        std::string byteString = hex.substr(i * 2, 2);
        buf[i] = (u8)strtol(byteString.c_str(), NULL, 16);
    }
}

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void parse_hash(const std::string& hash_line, pkzip_t& pkzip_struct) {
    std::vector<std::string> tokens = split(hash_line, '*');
    
    size_t pos = tokens[0].rfind('$');
    pkzip_struct.hash_count = std::stoi(tokens[0].substr(pos + 1));
    
    pkzip_struct.checksum_size = std::stoi(tokens[1]);
    pkzip_struct.version = std::stoi(tokens[2]);
    
    pkzip_struct.hash.data_type_enum = std::stoi(tokens[3]);
    pkzip_struct.hash.magic_type_enum = 0; 
    
    pkzip_struct.hash.compressed_length = hex2int(tokens[4]);
    pkzip_struct.hash.uncompressed_length = hex2int(tokens[5]);
    pkzip_struct.hash.crc32 = hex2int(tokens[6]);
    pkzip_struct.hash.offset = hex2int(tokens[7]);
    pkzip_struct.hash.additional_offset = hex2int(tokens[8]);
    pkzip_struct.hash.compression_type = std::stoi(tokens[9]);
    pkzip_struct.hash.data_length = hex2int(tokens[10]);
    pkzip_struct.hash.checksum_from_crc = hex2int(tokens[11]);
    pkzip_struct.hash.checksum_from_timestamp = 0; 

    std::string data_hex = tokens[12];
    if (!data_hex.empty() && data_hex.back() == '\n') data_hex.pop_back();
    if (!data_hex.empty() && data_hex.back() == '\r') data_hex.pop_back();

    std::memset(pkzip_struct.hash.data, 0, sizeof(pkzip_struct.hash.data));
    hex2bin(data_hex, (u8*)pkzip_struct.hash.data, sizeof(pkzip_struct.hash.data));
}

static size_t default_max_pw_bytes()
{
    // Typical wordlists (e.g. rockyou) rarely exceed this, and keeping it small
    // dramatically reduces H2D bandwidth. Override via ZIPCRACK_MAX_PW_BYTES.
    return 64;
}

// ---------------------- CPU Verification ---------------------------
size_t g_max_uncompressed_size = 100 * 1024 * 1024; // Default 100MB

bool try_unzip_with_password(const char * password, const char * zippath )
{
    if (zippath == nullptr || password == nullptr) return false;
    void *reader = nullptr;
    int32_t status = MZ_OK;

    // Avoid allocating the full uncompressed buffer for every candidate password.
    // Wrong passwords typically fail early; even for correct ones we can stream.
    std::vector<uint8_t> buffer(64 * 1024);
    
    reader = mz_zip_reader_create();
    if (reader == nullptr) return false;

    status = mz_zip_reader_open_file(reader, zippath);
    if (status != MZ_OK)
    {
        mz_zip_reader_delete(&reader);
        return false;
    }

    mz_zip_reader_set_password(reader, password);

    bool any_entry_ok = false;

    status = mz_zip_reader_goto_first_entry(reader);
    if (status == MZ_OK)
    {
        do
        {
            mz_zip_file *file_info = nullptr;
            status = mz_zip_reader_entry_get_info(reader, &file_info);
            
            if (status != MZ_OK || file_info == nullptr)
            {
                status = mz_zip_reader_goto_next_entry(reader);
                continue;
            }
            const char *name = file_info->filename ? file_info->filename : "";
            size_t nlen = strlen(name);
            int inferred_is_dir = (nlen > 0 && name[nlen - 1] == '/');

            if (inferred_is_dir)
            {
                status = mz_zip_reader_goto_next_entry(reader);
                continue;
            }

            status = mz_zip_reader_entry_open(reader);            
            if (status != MZ_OK)
            {
                status = mz_zip_reader_goto_next_entry(reader);
                continue;
            }

            const int64_t uncompressed_size = file_info->uncompressed_size;
            if (uncompressed_size < 0 || uncompressed_size > g_max_uncompressed_size)
            {
                fprintf(stderr, "Error: File '%s' is too large (%ld bytes). Max allowed is %lu bytes.\n", 
                        name, (long)uncompressed_size, g_max_uncompressed_size);
                mz_zip_reader_entry_close(reader);
                mz_zip_reader_close(reader);
                mz_zip_reader_delete(&reader);
                throw std::runtime_error("File too large");
            }

            int64_t total_read = 0;
            while (total_read < uncompressed_size)
            {
                const int64_t remaining = uncompressed_size - total_read;
                const int32_t want = static_cast<int32_t>(std::min<int64_t>(remaining, buffer.size()));

                int32_t rd = mz_zip_reader_entry_read(reader, buffer.data(), want);
                if (rd < 0) break;
                if (rd == 0) break; 
                total_read += rd;
            }

            int32_t close_status = mz_zip_reader_entry_close(reader);

            if (total_read == uncompressed_size && close_status == MZ_OK)
            {
                any_entry_ok = true;
                break; 
            }

            status = mz_zip_reader_goto_next_entry(reader);
        }
        while (status == MZ_OK);
    }

    mz_zip_reader_close(reader);
    mz_zip_reader_delete(&reader);

    return any_entry_ok;
}

#define BATCH_SIZE 100000

std::string get_hash_from_zip(const std::string& zip_path) {
    // Try local dependency first, then system path
    std::string zip2john_path = "./deps/john/run/zip2john";
    std::ifstream f(zip2john_path);
    if (!f.good()) {
        zip2john_path = "zip2john";
    }

    std::string cmd = zip2john_path + " " + zip_path;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    
    char buffer[128];
    std::string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    
    // zip2john might output filename first, e.g. "filename.zip:$pkzip$..."
    // We need to extract the hash part starting with $pkzip$
    size_t pos = result.find("$pkzip$");
    if (pos != std::string::npos) {
        // Find the end of the hash line (newline)
        size_t end_pos = result.find('\n', pos);
        if (end_pos != std::string::npos) {
            return result.substr(pos, end_pos - pos);
        }
        return result.substr(pos);
    }
    return "";
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <zip_file> <dict_file> [max_size_mb]" << std::endl;
        return 1;
    }

    std::string zip_file_path = argv[1];
    std::string dict_file_path = argv[2];

    const bool timing = (std::getenv("ZIPCRACK_TIMING") != nullptr);
    using clock = std::chrono::steady_clock;
    auto ms_since = [] (clock::duration d) -> double {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    double cpu_read_pack_ms = 0.0;
    double cpu_verify_ms = 0.0;
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;
    double kernel_ms = 0.0;
    u64 total_pw_tested = 0;
    u64 total_gpu_matches = 0;

    if (argc == 4) {
        try {
            size_t mb = std::stoul(argv[3]);
            g_max_uncompressed_size = mb * 1024 * 1024;
            std::cout << "Setting max uncompressed size to " << mb << " MB (" << g_max_uncompressed_size << " bytes)" << std::endl;
        } catch (...) {
            std::cerr << "Invalid max size argument. Using default." << std::endl;
        }
    }

    // 1. Generate Hash
    std::string hash_line = get_hash_from_zip(zip_file_path);
    if (hash_line.empty()) {
        std::cerr << "Error generating hash from zip file: " << zip_file_path << std::endl;
        return 1;
    }
    
    // std::cout << "Generated hash: " << hash_line << std::endl;

    pkzip_t h_pkzip{};
    parse_hash(hash_line, h_pkzip);

    // 2. Setup Device Memory for Hash
    pkzip_t *d_pkzip;
    cudaMalloc(&d_pkzip, sizeof(pkzip_t));
    cudaMemcpy(d_pkzip, &h_pkzip, sizeof(pkzip_t), cudaMemcpyHostToDevice);

    digest_t h_digest = {0}; 
    digest_t *d_digest;
    cudaMalloc(&d_digest, sizeof(digest_t));
    cudaMemcpy(d_digest, &h_digest, sizeof(digest_t), cudaMemcpyHostToDevice);

    // 3. Setup Device + Host Memory for Passwords and Results (double-buffered)
    constexpr int kBuffers = 2;

    size_t max_pw_bytes = default_max_pw_bytes();
    if (const char *env = std::getenv("ZIPCRACK_MAX_PW_BYTES"))
    {
        try
        {
            max_pw_bytes = std::stoul(env);
        }
        catch (...)
        {
            std::cerr << "Invalid ZIPCRACK_MAX_PW_BYTES; using default.\n";
            max_pw_bytes = default_max_pw_bytes();
        }
    }
    if (max_pw_bytes == 0) max_pw_bytes = default_max_pw_bytes();

    const u64 storage_cap_u64 = static_cast<u64>(BATCH_SIZE) * static_cast<u64>(max_pw_bytes);
    if (storage_cap_u64 > std::numeric_limits<u32>::max())
    {
        std::cerr << "ZIPCRACK_MAX_PW_BYTES too large for BATCH_SIZE (offsets use u32).\n";
        return 1;
    }
    const size_t pw_storage_capacity = static_cast<size_t>(storage_cap_u64);

    if (timing)
    {
        std::cerr << "Max password bytes: " << max_pw_bytes << "\n";
    }

    u8  *h_pw_storage[kBuffers] = {nullptr, nullptr};
    u32 *h_pw_offsets[kBuffers] = {nullptr, nullptr}; // length count+1
    u32 *h_match[kBuffers]      = {nullptr, nullptr};

    std::unique_ptr<u8[]>  h_pw_storage_pageable[kBuffers];
    std::unique_ptr<u32[]> h_pw_offsets_pageable[kBuffers];
    std::unique_ptr<u32[]> h_match_pageable[kBuffers];

    bool use_pinned = true;
    for (int i = 0; i < kBuffers; i++)
    {
        if (cudaMallocHost(&h_pw_storage[i], pw_storage_capacity) != cudaSuccess ||
            cudaMallocHost(&h_pw_offsets[i], (BATCH_SIZE + 1) * sizeof(u32)) != cudaSuccess ||
            cudaMallocHost(&h_match[i], BATCH_SIZE * sizeof(u32)) != cudaSuccess)
        {
            use_pinned = false;
            break;
        }
    }

    if (!use_pinned)
    {
        std::cerr << "Warning: cudaMallocHost failed; falling back to pageable host buffers (no H2D overlap)." << std::endl;

        for (int i = 0; i < kBuffers; i++)
        {
            if (h_pw_storage[i]) cudaFreeHost(h_pw_storage[i]);
            if (h_pw_offsets[i]) cudaFreeHost(h_pw_offsets[i]);
            if (h_match[i]) cudaFreeHost(h_match[i]);
            h_pw_storage[i] = nullptr;
            h_pw_offsets[i] = nullptr;
            h_match[i] = nullptr;
        }

        for (int i = 0; i < kBuffers; i++)
        {
            h_pw_storage_pageable[i] = std::make_unique<u8[]>(pw_storage_capacity);
            h_pw_offsets_pageable[i] = std::make_unique<u32[]>(BATCH_SIZE + 1);
            h_match_pageable[i] = std::make_unique<u32[]>(BATCH_SIZE);
            h_pw_storage[i] = h_pw_storage_pageable[i].get();
            h_pw_offsets[i] = h_pw_offsets_pageable[i].get();
            h_match[i] = h_match_pageable[i].get();
        }
    }

    u8  *d_pw_storage[kBuffers] = {nullptr, nullptr};
    u32 *d_pw_offsets[kBuffers] = {nullptr, nullptr};
    u32 *d_match[kBuffers]      = {nullptr, nullptr};

    for (int i = 0; i < kBuffers; i++)
    {
        cudaMalloc(&d_pw_storage[i], pw_storage_capacity);
        cudaMalloc(&d_pw_offsets[i], (BATCH_SIZE + 1) * sizeof(u32));
        cudaMalloc(&d_match[i], BATCH_SIZE * sizeof(u32));
    }

    cudaStream_t streams[kBuffers]{};
    for (int i = 0; i < kBuffers; i++)
    {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    cudaEvent_t ev_done[kBuffers]{};
    for (int i = 0; i < kBuffers; i++)
    {
        cudaEventCreateWithFlags(&ev_done[i], cudaEventDisableTiming);
    }

    cudaEvent_t ev_h2d_start[kBuffers]{}, ev_h2d_stop[kBuffers]{};
    cudaEvent_t ev_k_start[kBuffers]{}, ev_k_stop[kBuffers]{};
    cudaEvent_t ev_d2h_start[kBuffers]{}, ev_d2h_stop[kBuffers]{};

    if (timing)
    {
        for (int i = 0; i < kBuffers; i++)
        {
            cudaEventCreate(&ev_h2d_start[i]);
            cudaEventCreate(&ev_h2d_stop[i]);
            cudaEventCreate(&ev_k_start[i]);
            cudaEventCreate(&ev_k_stop[i]);
            cudaEventCreate(&ev_d2h_start[i]);
            cudaEventCreate(&ev_d2h_stop[i]);
        }
    }

    // 4. Read Dictionary and Run
    std::ifstream dict_file(dict_file_path);
    if (!dict_file.is_open()) {
        std::cerr << "Error opening dictionary file: " << dict_file_path << std::endl;
        return 1;
    }

    std::string pw_line;
    std::atomic<bool> found{false};
    std::string correct_password;
    std::atomic<bool> fatal_error{false};
    std::string fatal_error_msg;
    u64 skipped_too_long = 0;
    u32 pw_storage_bytes[kBuffers] = {0, 0};

    auto fill_batch = [&] (int buf) -> size_t {
        const auto t0 = clock::now();

        size_t count = 0;
        u32 offset = 0;
        h_pw_offsets[buf][0] = 0;

        while (count < BATCH_SIZE && std::getline(dict_file, pw_line))
        {
            if (!pw_line.empty() && pw_line.back() == '\r') pw_line.pop_back();

            if (pw_line.size() > max_pw_bytes)
            {
                skipped_too_long++;
                continue;
            }

            if (!pw_line.empty())
            {
                std::memcpy(h_pw_storage[buf] + offset, pw_line.data(), pw_line.size());
                offset += static_cast<u32>(pw_line.size());
            }

            h_pw_offsets[buf][count + 1] = offset;
            count++;
        }

        pw_storage_bytes[buf] = offset;
        cpu_read_pack_ms += ms_since(clock::now() - t0);
        return count;
    };

    auto launch_gpu = [&] (int buf, size_t count) {
        const int threadsPerBlock = 256;
        const int blocksPerGrid = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);

        if (timing) cudaEventRecord(ev_h2d_start[buf], streams[buf]);
        cudaMemcpyAsync(d_pw_storage[buf], h_pw_storage[buf], pw_storage_bytes[buf], cudaMemcpyHostToDevice, streams[buf]);
        cudaMemcpyAsync(d_pw_offsets[buf], h_pw_offsets[buf], (count + 1) * sizeof(u32), cudaMemcpyHostToDevice, streams[buf]);
        if (timing) cudaEventRecord(ev_h2d_stop[buf], streams[buf]);

        if (timing) cudaEventRecord(ev_k_start[buf], streams[buf]);
        m17200_sxx_cuda_optimized<<<blocksPerGrid, threadsPerBlock, 0, streams[buf]>>>(d_pkzip, d_digest, d_pw_storage[buf], d_pw_offsets[buf], (u32)count, d_match[buf]);
        if (timing) cudaEventRecord(ev_k_stop[buf], streams[buf]);

        if (timing) cudaEventRecord(ev_d2h_start[buf], streams[buf]);
        cudaMemcpyAsync(h_match[buf], d_match[buf], count * sizeof(u32), cudaMemcpyDeviceToHost, streams[buf]);
        if (timing) cudaEventRecord(ev_d2h_stop[buf], streams[buf]);

        cudaEventRecord(ev_done[buf], streams[buf]);
    };

    auto process_results = [&] (int buf, size_t count) {
        cudaEventSynchronize(ev_done[buf]);

        if (timing)
        {
            float ms = 0.0f;

            cudaEventElapsedTime(&ms, ev_h2d_start[buf], ev_h2d_stop[buf]);
            h2d_ms += ms;

            cudaEventElapsedTime(&ms, ev_k_start[buf], ev_k_stop[buf]);
            kernel_ms += ms;

            cudaEventElapsedTime(&ms, ev_d2h_start[buf], ev_d2h_stop[buf]);
            d2h_ms += ms;
        }

        std::vector<std::string> candidates;
        candidates.reserve(64);

        for (size_t i = 0; i < count; i++)
        {
            if (h_match[buf][i] != 1) continue;

            const u32 start = h_pw_offsets[buf][i];
            const u32 end   = h_pw_offsets[buf][i + 1];
            const u32 len   = end - start;

            std::string candidate;
            candidate.resize(len);
            if (len) std::memcpy(candidate.data(), h_pw_storage[buf] + start, len);
            candidates.push_back(std::move(candidate));
        }

        total_gpu_matches += candidates.size();

        if (!candidates.empty())
        {
            const auto v0 = clock::now();
            #pragma omp parallel for shared(found, correct_password, fatal_error, fatal_error_msg)
            for (size_t i = 0; i < candidates.size(); i++) {
                if (found.load(std::memory_order_relaxed) || fatal_error.load(std::memory_order_relaxed)) continue;

                try
                {
                    if (try_unzip_with_password(candidates[i].c_str(), zip_file_path.c_str())) {
                        #pragma omp critical
                        {
                            if (!found.load(std::memory_order_relaxed))
                            {
                                found.store(true, std::memory_order_relaxed);
                                correct_password = candidates[i];
                            }
                        }
                    }
                }
                catch (const std::runtime_error &e)
                {
                    #pragma omp critical
                    {
                        fatal_error.store(true, std::memory_order_relaxed);
                        fatal_error_msg = e.what();
                    }
                }
            }
            cpu_verify_ms += ms_since(clock::now() - v0);
        }
    };

    size_t count[kBuffers] = {0, 0};
    bool in_flight[kBuffers] = {false, false};
    bool eof = false;

    // Prime pipeline (up to 2 batches)
    for (int i = 0; i < kBuffers; i++)
    {
        count[i] = fill_batch(i);
        if (count[i] == 0)
        {
            eof = true;
            break;
        }
        launch_gpu(i, count[i]);
        in_flight[i] = true;
        total_pw_tested += count[i];
    }

    int proc = 0;
    while ((in_flight[0] || in_flight[1]) && !found.load(std::memory_order_relaxed) &&
           !fatal_error.load(std::memory_order_relaxed))
    {
        if (in_flight[proc])
        {
            process_results(proc, count[proc]);
            in_flight[proc] = false;
            if (found.load(std::memory_order_relaxed) || fatal_error.load(std::memory_order_relaxed)) break;
        }

        if (!eof && !fatal_error.load(std::memory_order_relaxed))
        {
            count[proc] = fill_batch(proc);
            if (count[proc] == 0)
            {
                eof = true;
            }
            else
            {
                launch_gpu(proc, count[proc]);
                in_flight[proc] = true;
                total_pw_tested += count[proc];
            }
        }

        proc = 1 - proc;
    }

    // Cleanup
    cudaDeviceSynchronize();

    if (timing)
    {
        for (int i = 0; i < kBuffers; i++)
        {
            cudaEventDestroy(ev_h2d_start[i]);
            cudaEventDestroy(ev_h2d_stop[i]);
            cudaEventDestroy(ev_k_start[i]);
            cudaEventDestroy(ev_k_stop[i]);
            cudaEventDestroy(ev_d2h_start[i]);
            cudaEventDestroy(ev_d2h_stop[i]);
        }
    }

    for (int i = 0; i < kBuffers; i++)
    {
        cudaEventDestroy(ev_done[i]);
        cudaStreamDestroy(streams[i]);
    }

    if (use_pinned)
    {
        for (int i = 0; i < kBuffers; i++)
        {
            cudaFreeHost(h_pw_storage[i]);
            cudaFreeHost(h_pw_offsets[i]);
            cudaFreeHost(h_match[i]);
        }
    }

    cudaFree(d_pkzip);
    cudaFree(d_digest);
    for (int i = 0; i < kBuffers; i++)
    {
        cudaFree(d_pw_storage[i]);
        cudaFree(d_pw_offsets[i]);
        cudaFree(d_match[i]);
    }

    if (fatal_error.load(std::memory_order_relaxed))
    {
        std::cerr << "Fatal Error: " << fatal_error_msg << std::endl;
        return 1;
    }

    if (timing)
    {
        std::cerr << "Timing summary:\n";
        std::cerr << "  tested:   " << total_pw_tested << "\n";
        std::cerr << "  matches:  " << total_gpu_matches << "\n";
        std::cerr << "  skipped:  " << skipped_too_long << " (len > " << max_pw_bytes << ")\n";
        std::cerr << "  cpu r/p:  " << std::fixed << std::setprecision(2) << cpu_read_pack_ms << " ms\n";
        std::cerr << "  cpu ver:  " << std::fixed << std::setprecision(2) << cpu_verify_ms << " ms\n";
        std::cerr << "  H2D:      " << std::fixed << std::setprecision(2) << h2d_ms << " ms\n";
        std::cerr << "  kernel:   " << std::fixed << std::setprecision(2) << kernel_ms << " ms\n";
        std::cerr << "  D2H:      " << std::fixed << std::setprecision(2) << d2h_ms << " ms\n";
    }

    if (found.load(std::memory_order_relaxed)) {
        std::cout << "FOUND PASSWORD: " << correct_password << std::endl;
        return 0;
    } else {
        std::cout << "Password not found." << std::endl;
        return 1;
    }
}
