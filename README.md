# CUDA Zip Cracker

A GPU-accelerated Zip password cracker using CUDA. This project implements a custom kernel to crack PKZIP-encrypted files, supporting large files (1GB+) which standard tools like Hashcat may fail to process.

## Features

- **CUDA Acceleration**: Uses NVIDIA GPUs for parallel password testing.
- **Large File Support**: Handles large zip archives that exceed the buffer limits of tools like `zip2john` and `hashcat`.
- **Minizip-ng Integration**: Uses `minizip-ng` for parsing zip file structures.

## Prerequisites

- NVIDIA GPU with CUDA support.
- CUDA Toolkit (nvcc).
- CMake (for building dependencies).
- Zlib.
- OpenSSL (for building John the Ripper).

## Building

1.  **Clone the repository with submodules**:
    ```bash
    git clone --recursive <repo_url>
    ```
    Or if you already cloned it:
    ```bash
    git submodule update --init --recursive
    ```

2.  **Build John the Ripper (zip2john)**:
    This project relies on `zip2john` to extract hashes. You can either install it system-wide or build the included submodule.

    ```bash
    cd deps/john/src
    ./configure && make -s clean && make -s
    cd ../../..
    ```

3.  **Build Project (and Minizip-ng)**:
    The `Makefile` automatically compiles `minizip-ng` (the library used for parsing zip files) and links it. You do not need to build `minizip-ng` manually.

    ```bash
    make
    ```

    This will produce the `final_project` executable.

4.  **Clean Build**:
    ```bash
    make clean
    ```
    ```

## Usage

```bash
./final_project <zip_file> <wordlist> [max_size_mb]
```

- `<zip_file>`: Path to the encrypted zip file.
- `<wordlist>`: Path to the password dictionary file (e.g., `rockyou.txt`).
- `[max_size_mb]`: (Optional) Maximum allowed uncompressed size in MB for verification. Defaults to 100MB.

## Example

```bash
./final_project test_gigantic.zip rockyou.txt 2000
```

## Performance Profiling

Enable a simple per-stage timing breakdown (CPU read/pack, H2D, kernel, D2H, CPU verify):

```bash
ZIPCRACK_TIMING=1 ./final_project <zip_file> <wordlist>
```

### Password Packing

To reduce H2D bandwidth, passwords are packed into a flat byte buffer plus per-password offsets before being copied to the GPU.
Very long passwords can increase buffer size and reduce overlap, so you can cap the maximum password length (longer lines are skipped):

```bash
ZIPCRACK_MAX_PW_BYTES=64 ./final_project <zip_file> <wordlist>
```

## Project Structure

- `final_project.cu`: Main host code, handles file I/O and orchestrates the cracking process.
- `kernel.cu`: CUDA kernel implementation for the cracking logic (CRC32, key updates, decryption checks).
- `deps/`: Contains the `minizip-ng` and `john` submodules.
