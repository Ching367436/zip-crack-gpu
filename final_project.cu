#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstring>
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

struct pw_t
{
    u32 i[64];
    u32 pw_len;
};

struct digest_t
{
    u32 digest_buf[4];
};

// External kernel declaration
__global__ void m17200_sxx_cuda_optimized(
    const pkzip_t *esalt_bufs,
    const digest_t *digests_buf,
    const pw_t *pws,
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

void pack_password(const std::string& pw, pw_t& pw_struct) {
    const size_t max_len = sizeof(pw_struct.i);
    const size_t len = std::min(pw.size(), max_len);

    pw_struct.pw_len = static_cast<u32>(len);

    if (len)
    {
        std::memcpy(pw_struct.i, pw.data(), len);

        // Only clear the unused bytes in the last word to keep the GPU-side
        // packed representation deterministic without memset()'ing 256 bytes.
        const size_t padded_len = (len + 3) & ~size_t(3);
        if (padded_len != len)
        {
            std::memset(reinterpret_cast<u8 *>(pw_struct.i) + len, 0, padded_len - len);
        }
    }
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

    // 3. Setup Device Memory for Passwords and Results
    pw_t *h_pws = new pw_t[BATCH_SIZE];
    u32 *h_match = new u32[BATCH_SIZE];

    pw_t *d_pws;
    u32 *d_match;
    cudaMalloc(&d_pws, BATCH_SIZE * sizeof(pw_t));
    cudaMalloc(&d_match, BATCH_SIZE * sizeof(u32));

    // 4. Read Dictionary and Run
    std::ifstream dict_file(dict_file_path);
    if (!dict_file.is_open()) {
        std::cerr << "Error opening dictionary file: " << dict_file_path << std::endl;
        return 1;
    }

    std::string pw_line;
    size_t batch_count = 0;
    bool found = false;
    std::string correct_password;

    cudaEvent_t ev_k_start{};
    cudaEvent_t ev_k_stop{};
    cudaEventCreate(&ev_k_start);
    cudaEventCreate(&ev_k_stop);

    auto batch_read_start = clock::now();

    auto run_gpu_filter = [&] (size_t count) {
        const auto t0 = clock::now();
        cudaMemcpy(d_pws, h_pws, count * sizeof(pw_t), cudaMemcpyHostToDevice);
        const auto t1 = clock::now();
        h2d_ms += ms_since(t1 - t0);

        const int threadsPerBlock = 256;
        const int blocksPerGrid = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);

        cudaEventRecord(ev_k_start);
        m17200_sxx_cuda_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_pkzip, d_digest, d_pws, (u32)count, d_match);
        cudaEventRecord(ev_k_stop);
        cudaEventSynchronize(ev_k_stop);

        if (timing)
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_k_start, ev_k_stop);
            kernel_ms += ms;
        }

        const auto t2 = clock::now();
        cudaMemcpy(h_match, d_match, count * sizeof(u32), cudaMemcpyDeviceToHost);
        const auto t3 = clock::now();
        d2h_ms += ms_since(t3 - t2);
    };

    while (std::getline(dict_file, pw_line) && !found) {
        if (!pw_line.empty() && pw_line.back() == '\r') pw_line.pop_back();

        pack_password(pw_line, h_pws[batch_count]);
        batch_count++;

        if (batch_count == BATCH_SIZE) {
            cpu_read_pack_ms += ms_since(clock::now() - batch_read_start);

            // GPU Filter
            run_gpu_filter(BATCH_SIZE);
            total_pw_tested += BATCH_SIZE;

            // Collect candidates
            std::vector<u32> candidates;
            for (size_t i = 0; i < BATCH_SIZE; i++) {
                if (h_match[i] == 1) {
                    candidates.push_back((u32)i);
                }
            }
            total_gpu_matches += candidates.size();

            // CPU Parallel Verification
            if (!candidates.empty()) {
                try {
                    const auto v0 = clock::now();
                    #pragma omp parallel for shared(found, correct_password)
                    for (size_t i = 0; i < candidates.size(); i++) {
                        if (found) continue; // Early exit if found by another thread

                        const pw_t &pw = h_pws[candidates[i]];
                        std::string candidate;
                        candidate.resize(pw.pw_len);
                        if (pw.pw_len) std::memcpy(candidate.data(), pw.i, pw.pw_len);

                        if (try_unzip_with_password(candidate.c_str(), zip_file_path.c_str())) {
                            #pragma omp critical
                            {
                                found = true;
                                correct_password = candidate;
                            }
                        }
                    }
                    cpu_verify_ms += ms_since(clock::now() - v0);
                } catch (const std::runtime_error& e) {
                    std::cerr << "Fatal Error: " << e.what() << std::endl;
                    return 1;
                }
            }

            batch_count = 0;
            batch_read_start = clock::now();
        }
    }

    // Process remaining
    if (batch_count && !found) {
        const size_t count = batch_count;
        cpu_read_pack_ms += ms_since(clock::now() - batch_read_start);

        run_gpu_filter(count);
        total_pw_tested += count;

        std::vector<u32> candidates;
        for (size_t i = 0; i < count; i++) {
            if (h_match[i] == 1) {
                candidates.push_back((u32)i);
            }
        }
        total_gpu_matches += candidates.size();

        if (!candidates.empty()) {
            try {
                const auto v0 = clock::now();
                #pragma omp parallel for shared(found, correct_password)
                for (size_t i = 0; i < candidates.size(); i++) {
                    if (found) continue;

                    const pw_t &pw = h_pws[candidates[i]];
                    std::string candidate;
                    candidate.resize(pw.pw_len);
                    if (pw.pw_len) std::memcpy(candidate.data(), pw.i, pw.pw_len);

                    if (try_unzip_with_password(candidate.c_str(), zip_file_path.c_str())) {
                        #pragma omp critical
                        {
                            found = true;
                            correct_password = candidate;
                        }
                    }
                }
                cpu_verify_ms += ms_since(clock::now() - v0);
            } catch (const std::runtime_error& e) {
                std::cerr << "Fatal Error: " << e.what() << std::endl;
                return 1;
            }
        }
    }

    cudaEventDestroy(ev_k_start);
    cudaEventDestroy(ev_k_stop);

    // Cleanup
    delete[] h_pws;
    delete[] h_match;
    cudaFree(d_pkzip);
    cudaFree(d_digest);
    cudaFree(d_pws);
    cudaFree(d_match);

    if (timing)
    {
        std::cerr << "Timing summary:\n";
        std::cerr << "  tested:   " << total_pw_tested << "\n";
        std::cerr << "  matches:  " << total_gpu_matches << "\n";
        std::cerr << "  cpu r/p:  " << std::fixed << std::setprecision(2) << cpu_read_pack_ms << " ms\n";
        std::cerr << "  cpu ver:  " << std::fixed << std::setprecision(2) << cpu_verify_ms << " ms\n";
        std::cerr << "  H2D:      " << std::fixed << std::setprecision(2) << h2d_ms << " ms\n";
        std::cerr << "  kernel:   " << std::fixed << std::setprecision(2) << kernel_ms << " ms\n";
        std::cerr << "  D2H:      " << std::fixed << std::setprecision(2) << d2h_ms << " ms\n";
    }

    if (found) {
        std::cout << "FOUND PASSWORD: " << correct_password << std::endl;
        return 0;
    } else {
        std::cout << "Password not found." << std::endl;
        return 1;
    }
}
