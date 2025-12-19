#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
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
    u32 data[512]; 
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

            std::vector<uint8_t> buffer;
            buffer.resize(static_cast<size_t>(uncompressed_size));

            int64_t total_read = 0;
            while (total_read < uncompressed_size)
            {
                int32_t rd = mz_zip_reader_entry_read(reader,
                                                      buffer.data() + total_read,
                                                      static_cast<int32_t>(uncompressed_size - total_read));
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
    std::vector<std::string> current_batch_pws;
    bool found = false;
    std::string correct_password;

    while (std::getline(dict_file, pw_line) && !found) {
        if (!pw_line.empty() && pw_line.back() == '\r') pw_line.pop_back();
        
        current_batch_pws.push_back(pw_line);

        if (current_batch_pws.size() == BATCH_SIZE) {
            // GPU Filter
            for (size_t i = 0; i < BATCH_SIZE; i++) {
                pack_password(current_batch_pws[i], h_pws[i]);
            }

            cudaMemcpy(d_pws, h_pws, BATCH_SIZE * sizeof(pw_t), cudaMemcpyHostToDevice);
            
            int threadsPerBlock = 256;
            int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;
            
            m17200_sxx_cuda_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_pkzip, d_digest, d_pws, BATCH_SIZE, d_match);
            cudaDeviceSynchronize();

            cudaMemcpy(h_match, d_match, BATCH_SIZE * sizeof(u32), cudaMemcpyDeviceToHost);

            // Collect candidates
            std::vector<std::string> candidates;
            for (size_t i = 0; i < BATCH_SIZE; i++) {
                if (h_match[i] == 1) {
                    candidates.push_back(current_batch_pws[i]);
                }
            }

            // CPU Parallel Verification
            if (!candidates.empty()) {
                try {
                    #pragma omp parallel for shared(found, correct_password)
                    for (size_t i = 0; i < candidates.size(); i++) {
                        if (found) continue; // Early exit if found by another thread
                        
                        if (try_unzip_with_password(candidates[i].c_str(), zip_file_path.c_str())) {
                            #pragma omp critical
                            {
                                found = true;
                                correct_password = candidates[i];
                            }
                        }
                    }
                } catch (const std::runtime_error& e) {
                    std::cerr << "Fatal Error: " << e.what() << std::endl;
                    return 1;
                }
            }

            current_batch_pws.clear();
        }
    }

    // Process remaining
    if (!current_batch_pws.empty() && !found) {
        size_t count = current_batch_pws.size();
        for (size_t i = 0; i < count; i++) {
            pack_password(current_batch_pws[i], h_pws[i]);
        }

        cudaMemcpy(d_pws, h_pws, count * sizeof(pw_t), cudaMemcpyHostToDevice);
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
        
        m17200_sxx_cuda_optimized<<<blocksPerGrid, threadsPerBlock>>>(d_pkzip, d_digest, d_pws, count, d_match);
        cudaDeviceSynchronize();

        cudaMemcpy(h_match, d_match, count * sizeof(u32), cudaMemcpyDeviceToHost);

        std::vector<std::string> candidates;
        for (size_t i = 0; i < count; i++) {
            if (h_match[i] == 1) {
                candidates.push_back(current_batch_pws[i]);
            }
        }

        if (!candidates.empty()) {
            try {
                #pragma omp parallel for shared(found, correct_password)
                for (size_t i = 0; i < candidates.size(); i++) {
                    if (found) continue;
                    
                    if (try_unzip_with_password(candidates[i].c_str(), zip_file_path.c_str())) {
                        #pragma omp critical
                        {
                            found = true;
                            correct_password = candidates[i];
                        }
                    }
                }
            } catch (const std::runtime_error& e) {
                std::cerr << "Fatal Error: " << e.what() << std::endl;
                return 1;
            }
        }
    }

    // Cleanup
    delete[] h_pws;
    delete[] h_match;
    cudaFree(d_pkzip);
    cudaFree(d_digest);
    cudaFree(d_pws);
    cudaFree(d_match);

    if (found) {
        std::cout << "FOUND PASSWORD: " << correct_password << std::endl;
        return 0;
    } else {
        std::cout << "Password not found." << std::endl;
        return 1;
    }
}
