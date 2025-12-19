#include <stdint.h>
#include <cuda_runtime.h>

// ---------------------- Type aliases ----------------------
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;

// ---------------------- Constants -------------------------
// Only need indices 0..8:
// - 3 x u32 (12 bytes) PKZIP header
// - 6 x u32 (24 bytes) encrypted payload for inflate checks
#define MAX_LOCAL 9

#define MSB(x) ((x) >> 24)
#define CONST_PK 0x08088405

// ---------------------- CRC32 table -----------------------
__device__ __align__(16) const u32 crc32tab[256] = {
    0x00000000,0x77073096,0xee0e612c,0x990951ba,0x076dc419,0x706af48f,0xe963a535,0x9e6495a3,
    0x0edb8832,0x79dcb8a4,0xe0d5e91e,0x97d2d988,0x09b64c2b,0x7eb17cbd,0xe7b82d07,0x90bf1d91,
    0x1db71064,0x6ab020f2,0xf3b97148,0x84be41de,0x1adad47d,0x6ddde4eb,0xf4d4b551,0x83d385c7,
    0x136c9856,0x646ba8c0,0xfd62f97a,0x8a65c9ec,0x14015c4f,0x63066cd9,0xfa0f3d63,0x8d080df5,
    0x3b6e20c8,0x4c69105e,0xd56041e4,0xa2677172,0x3c03e4d1,0x4b04d447,0xd20d85fd,0xa50ab56b,
    0x35b5a8fa,0x42b2986c,0xdbbbc9d6,0xacbcf940,0x32d86ce3,0x45df5c75,0xdcd60dcf,0xabd13d59,
    0x26d930ac,0x51de003a,0xc8d75180,0xbfd06116,0x21b4f4b5,0x56b3c423,0xcfba9599,0xb8bda50f,
    0x2802b89e,0x5f058808,0xc60cd9b2,0xb10be924,0x2f6f7c87,0x58684c11,0xc1611dab,0xb6662d3d,
    0x76dc4190,0x01db7106,0x98d220bc,0xefd5102a,0x71b18589,0x06b6b51f,0x9fbfe4a5,0xe8b8d433,
    0x7807c9a2,0x0f00f934,0x9609a88e,0xe10e9818,0x7f6a0dbb,0x086d3d2d,0x91646c97,0xe6635c01,
    0x6b6b51f4,0x1c6c6162,0x856530d8,0xf262004e,0x6c0695ed,0x1b01a57b,0x8208f4c1,0xf50fc457,
    0x65b0d9c6,0x12b7e950,0x8bbeb8ea,0xfcb9887c,0x62dd1ddf,0x15da2d49,0x8cd37cf3,0xfbd44c65,
    0x4db26158,0x3ab551ce,0xa3bc0074,0xd4bb30e2,0x4adfa541,0x3dd895d7,0xa4d1c46d,0xd3d6f4fb,
    0x4369e96a,0x346ed9fc,0xad678846,0xda60b8d0,0x44042d73,0x33031de5,0xaa0a4c5f,0xdd0d7cc9,
    0x5005713c,0x270241aa,0xbe0b1010,0xc90c2086,0x5768b525,0x206f85b3,0xb966d409,0xce61e49f,
    0x5edef90e,0x29d9c998,0xb0d09822,0xc7d7a8b4,0x59b33d17,0x2eb40d81,0xb7bd5c3b,0xc0ba6cad,
    0xedb88320,0x9abfb3b6,0x03b6e20c,0x74b1d29a,0xead54739,0x9dd277af,0x04db2615,0x73dc1683,
    0xe3630b12,0x94643b84,0x0d6d6a3e,0x7a6a5aa8,0xe40ecf0b,0x9309ff9d,0x0a00ae27,0x7d079eb1,
    0xf00f9344,0x8708a3d2,0x1e01f268,0x6906c2fe,0xf762575d,0x806567cb,0x196c3671,0x6e6b06e7,
    0xfed41b76,0x89d32be0,0x10da7a5a,0x67dd4acc,0xf9b9df6f,0x8ebeeff9,0x17b7be43,0x60b08ed5,
    0xd6d6a3e8,0xa1d1937e,0x38d8c2c4,0x4fdff252,0xd1bb67f1,0xa6bc5767,0x3fb506dd,0x48b2364b,
    0xd80d2bda,0xaf0a1b4c,0x36034af6,0x41047a60,0xdf60efc3,0xa867df55,0x316e8eef,0x4669be79,
    0xcb61b38c,0xbc66831a,0x256fd2a0,0x5268e236,0xcc0c7795,0xbb0b4703,0x220216b9,0x5505262f,
    0xc5ba3bbe,0xb2bd0b28,0x2bb45a92,0x5cb36a04,0xc2d7ffa7,0xb5d0cf31,0x2cd99e8b,0x5bdeae1d,
    0x9b64c2b0,0xec63f226,0x756aa39c,0x026d930a,0x9c0906a9,0xeb0e363f,0x72076785,0x05005713,
    0x95bf4a82,0xe2b87a14,0x7bb12bae,0x0cb61b38,0x92d28e9b,0xe5d5be0d,0x7cdcefb7,0x0bdbdf21,
    0x86d3d2d4,0xf1d4e242,0x68ddb3f8,0x1fda836e,0x81be16cd,0xf6b9265b,0x6fb077e1,0x18b74777,
    0x88085ae6,0xff0f6a70,0x66063bca,0x11010b5c,0x8f659eff,0xf862ae69,0x616bffd3,0x166ccf45,
    0xa00ae278,0xd70dd2ee,0x4e048354,0x3903b3c2,0xa7672661,0xd06016f7,0x4969474d,0x3e6e77db,
    0xaed16a4a,0xd9d65adc,0x40df0b66,0x37d83bf0,0xa9bcae53,0xdebb9ec5,0x47b2cf7f,0x30b5ffe9,
    0xbdbdf21c,0xcabac28a,0x53b39330,0x24b4a3a6,0xbad03605,0xcdd70693,0x54de5729,0x23d967bf,
    0xb3667a2e,0xc4614ab8,0x5d681b02,0x2a6f2b94,0xb40bbe37,0xc30c8ea1,0x5a05df1b,0x2d02ef8d
};

// ---------------------- Helpers ---------------------------
__device__ __forceinline__ u32 crc32_device(u32 x, u32 c)
{
    return (x >> 8) ^ __ldg(&crc32tab[(x ^ c) & 0xff]);
}

__device__ __forceinline__ void update_key012(u32 &k0, u32 &k1, u32 &k2, u32 c)
{
    k0 = crc32_device(k0, c);
    k1 = (k1 + (k0 & 0xff)) * CONST_PK + 1;
    k2 = crc32_device(k2, MSB(k1));
}

__device__ __forceinline__ void update_key3(u32 k2, u32 &k3)
{
    const u32 temp = (k2 & 0xffff) | 3;
    k3 = ((temp * (temp ^ 1)) >> 8) & 0xff;
}

__device__ __forceinline__ u32 unpack_v8a_from_v32(u32 v) { return (v >> 0) & 0xff; }
__device__ __forceinline__ u32 unpack_v8b_from_v32(u32 v) { return (v >> 8) & 0xff; }
__device__ __forceinline__ u32 unpack_v8c_from_v32(u32 v) { return (v >> 16) & 0xff; }
__device__ __forceinline__ u32 unpack_v8d_from_v32(u32 v) { return (v >> 24) & 0xff; }

__device__ __forceinline__ u32 get_byte_24 (const u32 w0, const u32 w1, const u32 w2, const u32 w3, const u32 w4, const u32 w5, const u32 idx)
{
    const u32 shift = (idx & 3u) << 3;

    switch (idx >> 2)
    {
      case 0: return (w0 >> shift) & 0xff;
      case 1: return (w1 >> shift) & 0xff;
      case 2: return (w2 >> shift) & 0xff;
      case 3: return (w3 >> shift) & 0xff;
      case 4: return (w4 >> shift) & 0xff;
      default: return (w5 >> shift) & 0xff;
    }
}

// ---------------------- Structs ---------------------------
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
    u32 data[10]; // Kept original size for struct layout compatibility, but we won't load all of it
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

// ---------------------- Huffman tables --------------------
struct code
{
    u8  op;
    u8  bits;
    u16 val;
};

__device__ __constant__ code lenfix[512] = {
    {96,7,0},{0,8,80},{0,8,16},{20,8,115},{18,7,31},{0,8,112},{0,8,48},
    {0,9,192},{16,7,10},{0,8,96},{0,8,32},{0,9,160},{0,8,0},{0,8,128},
    {0,8,64},{0,9,224},{16,7,6},{0,8,88},{0,8,24},{0,9,144},{19,7,59},
    {0,8,120},{0,8,56},{0,9,208},{17,7,17},{0,8,104},{0,8,40},{0,9,176},
    {0,8,8},{0,8,136},{0,8,72},{0,9,240},{16,7,4},{0,8,84},{0,8,20},
    {21,8,227},{19,7,43},{0,8,116},{0,8,52},{0,9,200},{17,7,13},{0,8,100},
    {0,8,36},{0,9,168},{0,8,4},{0,8,132},{0,8,68},{0,9,232},{16,7,8},
    {0,8,92},{0,8,28},{0,9,152},{20,7,83},{0,8,124},{0,8,60},{0,9,216},
    {18,7,23},{0,8,108},{0,8,44},{0,9,184},{0,8,12},{0,8,140},{0,8,76},
    {0,9,248},{16,7,3},{0,8,82},{0,8,18},{21,8,163},{19,7,35},{0,8,114},
    {0,8,50},{0,9,196},{17,7,11},{0,8,98},{0,8,34},{0,9,164},{0,8,2},
    {0,8,130},{0,8,66},{0,9,228},{16,7,7},{0,8,90},{0,8,26},{0,9,148},
    {20,7,67},{0,8,122},{0,8,58},{0,9,212},{18,7,19},{0,8,106},{0,8,42},
    {0,9,180},{0,8,10},{0,8,138},{0,8,74},{0,9,244},{16,7,5},{0,8,86},
    {0,8,22},{64,8,0},{19,7,51},{0,8,118},{0,8,54},{0,9,204},{17,7,15},
    {0,8,102},{0,8,38},{0,9,172},{0,8,6},{0,8,134},{0,8,70},{0,9,236},
    {16,7,9},{0,8,94},{0,8,30},{0,9,156},{20,7,99},{0,8,126},{0,8,62},
    {0,9,220},{18,7,27},{0,8,110},{0,8,46},{0,9,188},{0,8,14},{0,8,142},
    {0,8,78},{0,9,252},{96,7,0},{0,8,81},{0,8,17},{21,8,131},{18,7,31},
    {0,8,113},{0,8,49},{0,9,194},{16,7,10},{0,8,97},{0,8,33},{0,9,162},
    {0,8,1},{0,8,129},{0,8,65},{0,9,226},{16,7,6},{0,8,89},{0,8,25},
    {0,9,146},{19,7,59},{0,8,121},{0,8,57},{0,9,210},{17,7,17},{0,8,105},
    {0,8,41},{0,9,178},{0,8,9},{0,8,137},{0,8,73},{0,9,242},{16,7,4},
    {0,8,85},{0,8,21},{16,8,258},{19,7,43},{0,8,117},{0,8,53},{0,9,202},
    {17,7,13},{0,8,101},{0,8,37},{0,9,170},{0,8,5},{0,8,133},{0,8,69},
    {0,9,234},{16,7,8},{0,8,93},{0,8,29},{0,9,154},{20,7,83},{0,8,125},
    {0,8,61},{0,9,218},{18,7,23},{0,8,109},{0,8,45},{0,9,186},{0,8,13},
    {0,8,141},{0,8,77},{0,9,250},{16,7,3},{0,8,83},{0,8,19},{21,8,195},
    {19,7,35},{0,8,115},{0,8,51},{0,9,198},{17,7,11},{0,8,99},{0,8,35},
    {0,9,166},{0,8,3},{0,8,131},{0,8,67},{0,9,230},{16,7,7},{0,8,91},
    {0,8,27},{0,9,150},{20,7,67},{0,8,123},{0,8,59},{0,9,214},{18,7,19},
    {0,8,107},{0,8,43},{0,9,182},{0,8,11},{0,8,139},{0,8,75},{0,9,246},
    {16,7,5},{0,8,87},{0,8,23},{64,8,0},{19,7,51},{0,8,119},{0,8,55},
    {0,9,206},{17,7,15},{0,8,103},{0,8,39},{0,9,174},{0,8,7},{0,8,135},
    {0,8,71},{0,9,238},{16,7,9},{0,8,95},{0,8,31},{0,9,158},{20,7,99},
    {0,8,127},{0,8,63},{0,9,222},{18,7,27},{0,8,111},{0,8,47},{0,9,190},
    {0,8,15},{0,8,143},{0,8,79},{0,9,254},{96,7,0},{0,8,80},{0,8,16},
    {20,8,115},{18,7,31},{0,8,112},{0,8,48},{0,9,193},{16,7,10},{0,8,96},
    {0,8,32},{0,9,161},{0,8,0},{0,8,128},{0,8,64},{0,9,225},{16,7,6},
    {0,8,88},{0,8,24},{0,9,145},{19,7,59},{0,8,120},{0,8,56},{0,9,209},
    {17,7,17},{0,8,104},{0,8,40},{0,9,177},{0,8,8},{0,8,136},{0,8,72},
    {0,9,241},{16,7,4},{0,8,84},{0,8,20},{21,8,227},{19,7,43},{0,8,116},
    {0,8,52},{0,9,201},{17,7,13},{0,8,100},{0,8,36},{0,9,169},{0,8,4},
    {0,8,132},{0,8,68},{0,9,233},{16,7,8},{0,8,92},{0,8,28},{0,9,153},
    {20,7,83},{0,8,124},{0,8,60},{0,9,217},{18,7,23},{0,8,108},{0,8,44},
    {0,9,185},{0,8,12},{0,8,140},{0,8,76},{0,9,249},{16,7,3},{0,8,82},
    {0,8,18},{21,8,163},{19,7,35},{0,8,114},{0,8,50},{0,9,197},{17,7,11},
    {0,8,98},{0,8,34},{0,9,165},{0,8,2},{0,8,130},{0,8,66},{0,9,229},
    {16,7,7},{0,8,90},{0,8,26},{0,9,149},{20,7,67},{0,8,122},{0,8,58},
    {0,9,213},{18,7,19},{0,8,106},{0,8,42},{0,9,181},{0,8,10},{0,8,138},
    {0,8,74},{0,9,245},{16,7,5},{0,8,86},{0,8,22},{64,8,0},{19,7,51},
    {0,8,118},{0,8,54},{0,9,205},{17,7,15},{0,8,102},{0,8,38},{0,9,173},
    {0,8,6},{0,8,134},{0,8,70},{0,9,237},{16,7,9},{0,8,94},{0,8,30},
    {0,9,157},{20,7,99},{0,8,126},{0,8,62},{0,9,221},{18,7,27},{0,8,110},
    {0,8,46},{0,9,189},{0,8,14},{0,8,142},{0,8,78},{0,9,253},{96,7,0},
    {0,8,81},{0,8,17},{21,8,131},{18,7,31},{0,8,113},{0,8,49},{0,9,195},
    {16,7,10},{0,8,97},{0,8,33},{0,9,163},{0,8,1},{0,8,129},{0,8,65},
    {0,9,227},{16,7,6},{0,8,89},{0,8,25},{0,9,147},{19,7,59},{0,8,121},
    {0,8,57},{0,9,211},{17,7,17},{0,8,105},{0,8,41},{0,9,179},{0,8,9},
    {0,8,137},{0,8,73},{0,9,243},{16,7,4},{0,8,85},{0,8,21},{16,8,258},
    {19,7,43},{0,8,117},{0,8,53},{0,9,203},{17,7,13},{0,8,101},{0,8,37},
    {0,9,171},{0,8,5},{0,8,133},{0,8,69},{0,9,235},{16,7,8},{0,8,93},
    {0,8,29},{0,9,155},{20,7,83},{0,8,125},{0,8,61},{0,9,219},{18,7,23},
    {0,8,109},{0,8,45},{0,9,187},{0,8,13},{0,8,141},{0,8,77},{0,9,251},
    {16,7,3},{0,8,83},{0,8,19},{21,8,195},{19,7,35},{0,8,115},{0,8,51},
    {0,9,199},{17,7,11},{0,8,99},{0,8,35},{0,9,167},{0,8,3},{0,8,131},
    {0,8,67},{0,9,231},{16,7,7},{0,8,91},{0,8,27},{0,9,151},{20,7,67},
    {0,8,123},{0,8,59},{0,9,215},{18,7,19},{0,8,107},{0,8,43},{0,9,183},
    {0,8,11},{0,8,139},{0,8,75},{0,9,247},{16,7,5},{0,8,87},{0,8,23},
    {64,8,0},{19,7,51},{0,8,119},{0,8,55},{0,9,207},{17,7,15},{0,8,103},
    {0,8,39},{0,9,175},{0,8,7},{0,8,135},{0,8,71},{0,9,239},{16,7,9},
    {0,8,95},{0,8,31},{0,9,159},{20,7,99},{0,8,127},{0,8,63},{0,9,223},
    {18,7,27},{0,8,111},{0,8,47},{0,9,191},{0,8,15},{0,8,143},{0,8,79},
    {0,9,255}
};

__device__ __constant__ code distfix[32] = {
    {16,5,1},{23,5,257},{19,5,17},{27,5,4097},{17,5,5},{25,5,1025},
    {21,5,65},{29,5,16385},{16,5,3},{24,5,513},{20,5,33},{28,5,8193},
    {18,5,9},{26,5,2049},{22,5,129},{64,5,0},{16,5,2},{23,5,385},
    {19,5,25},{27,5,6145},{17,5,7},{25,5,1537},{21,5,97},{29,5,24577},
    {16,5,4},{24,5,769},{20,5,49},{28,5,12289},{18,5,13},{26,5,3073},
    {22,5,193},{64,5,0}
};

// ---------------------- Check helpers ---------------------
__device__ int check_inflate_code2 (const u32 w0, const u32 w1, const u32 w2, const u32 w3, const u32 w4, const u32 w5)
{
    u32 bits, hold, thisget, have;
    int left;
    u32 ncode;
    u32 count1 = 0;
    u32 count2 = 0;
    u32 count3 = 0;
    u32 count4 = 0;
    u32 count5 = 0;
    u32 count6 = 0;
    u32 count7 = 0;

    u32 pos = 0;
    hold = get_byte_24(w0, w1, w2, w3, w4, w5, 0)
         + (get_byte_24(w0, w1, w2, w3, w4, w5, 1) << 8)
         + (get_byte_24(w0, w1, w2, w3, w4, w5, 2) << 16)
         + (get_byte_24(w0, w1, w2, w3, w4, w5, 3) << 24);
    pos = 3;
    hold >>= 3;

    if (257 + (hold & 0x1F) > 286)
    {
      return 0;
    }
    hold >>= 5;
    if (1 + (hold & 0x1F) > 30)
    {
      return 0;
    }
    hold >>= 5;
    ncode = 4 + (hold & 0xF);
    hold >>= 4;

    hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << 15;
    hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << 23;
    bits = 31;

    have = 0;

    for (;;)
    {
      if (have + 7 > ncode)
      {
        thisget = ncode - have;
      }
      else
      {
        thisget = 7;
      }
      have += thisget;
      bits -= thisget * 3;
      while (thisget--)
      {
        switch (hold & 7)
        {
          case 1: ++count1; break;
          case 2: ++count2; break;
          case 3: ++count3; break;
          case 4: ++count4; break;
          case 5: ++count5; break;
          case 6: ++count6; break;
          case 7: ++count7; break;
          default: break;
        }
        hold >>= 3;
      }
      if (have == ncode)
      {
        break;
      }
      hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
      bits += 8;
      hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
      bits += 8;
    }
    if ((count1 | count2 | count3 | count4 | count5 | count6 | count7) == 0)
    {
      return 0;
    }

    left = 1;
    left = (left << 1) - (int) count1; if (left < 0) return 0;
    left = (left << 1) - (int) count2; if (left < 0) return 0;
    left = (left << 1) - (int) count3; if (left < 0) return 0;
    left = (left << 1) - (int) count4; if (left < 0) return 0;
    left = (left << 1) - (int) count5; if (left < 0) return 0;
    left = (left << 1) - (int) count6; if (left < 0) return 0;
    left = (left << 1) - (int) count7; if (left < 0) return 0;
    if (left > 0)
    {
      return 0;
    }

    return 1;
}

__device__ int check_inflate_code1 (const u32 w0, const u32 w1, const u32 w2, const u32 w3, const u32 w4, const u32 w5, int left)
{
    u32 whave = 0, op, bits, hold, len;
    code here1;

    u32 pos = 0;
    hold = get_byte_24(w0, w1, w2, w3, w4, w5, 0)
         + (get_byte_24(w0, w1, w2, w3, w4, w5, 1) << 8)
         + (get_byte_24(w0, w1, w2, w3, w4, w5, 2) << 16)
         + (get_byte_24(w0, w1, w2, w3, w4, w5, 3) << 24);
    pos = 3;
    left -= 4;
    hold >>= 3;
    bits = 32 - 3;
    for (;;)
    {
      if (bits < 15)
        {
          if (left < 2)
          {
            return 1;
          }
          left -= 2;
          hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
          bits += 8;
          hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
          bits += 8;
        }
        here1 = lenfix[hold & 0x1FF];
        op = (unsigned)(here1.bits);
        hold >>= op;
      bits -= op;
      op = (unsigned)(here1.op);
      if (op == 0)
      {
        ++whave;
      }
      else if (op & 16)
      {
        len = (unsigned)(here1.val);
        op &= 15;
        if (op)
        {
          if (bits < op)
          {
            if (!left)
            {
              return 1;
            }
            --left;
            hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
            bits += 8;
          }
          len += (unsigned)hold & ((1U << op) - 1);
          hold >>= op;
          bits -= op;
        }
        if (bits < 15)
        {
          if (left < 2)
          {
            return 1;
          }
          left -= 2;
          hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
          bits += 8;
          hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
          bits += 8;
        }
        code here2 = distfix[hold & 0x1F];
        op = (unsigned)(here2.bits);
        hold >>= op;
        bits -= op;
        op = (unsigned)(here2.op);
        if (op & 16)
        {
          u32 dist = (unsigned)(here2.val);
          op &= 15;
          if (bits < op)
          {
            if (!left)
            {
              return 1;
            }
            --left;
            hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
            bits += 8;
            if (bits < op)
            {
              if (!left)
              {
                return 1;
              }
              --left;
              hold += get_byte_24(w0, w1, w2, w3, w4, w5, ++pos) << bits;
              bits += 8;
            }
          }
          dist += (unsigned)hold & ((1U << op) - 1);
          if (dist > whave)
          {
            return 0;
          }
          hold >>= op;
          bits -= op;

          whave += len;
        }
        else
        {
          return 0;
        }
      }
      else if (op & 32)
      {
        if (left == 0)
        {
          return 1;
        }
        return 0;
      }
      else
      {
        return 0;
      }
    }
}

// ---------------------- Kernels ---------------------------
__global__ void m17200_sxx_cuda_optimized(
    const pkzip_t *__restrict__ esalt_bufs,
    const digest_t *__restrict__ digests_buf,
    const pw_t *__restrict__ pws,
    u32 gid_cnt,
    u32 *__restrict__ match_out)
{
    const u32 gid = (u32)(blockIdx.x * blockDim.x + threadIdx.x);
    const u32 lid = threadIdx.x;
    const u32 lsz = blockDim.x;

    // Only load the words we actually use for early checks.
    __shared__ u32 l_data[MAX_LOCAL];
    for (u32 i = lid; i < MAX_LOCAL; i += lsz)
    {
      l_data[i] = esalt_bufs[0].hash.data[i];
    }

    __syncthreads();

    if (gid >= gid_cnt) return;

    // Initialize match_out to 0
    match_out[gid] = 0;

    const u32 checksum_size           = esalt_bufs[0].checksum_size;
    const u32 checksum_from_crc       = esalt_bufs[0].hash.checksum_from_crc;
    const u32 checksum_from_timestamp = esalt_bufs[0].hash.checksum_from_timestamp;
    const u32 data_length             = esalt_bufs[0].hash.data_length;

    // Removed apply_rules loop and logic

    u32 key0 = 0x12345678;
    u32 key1 = 0x23456789;
    u32 key2 = 0x34567890;

    const pw_t *pw = pws + gid;
    const u32 pw_len = pw->pw_len;

    for (u32 j = 0; j < (pw_len >> 2); j++)
    {
      const u32 w = pw->i[j];

      update_key012(key0, key1, key2, unpack_v8a_from_v32(w));
      update_key012(key0, key1, key2, unpack_v8b_from_v32(w));
      update_key012(key0, key1, key2, unpack_v8c_from_v32(w));
      update_key012(key0, key1, key2, unpack_v8d_from_v32(w));
    }

    const u32 rem = pw_len & 3;

    if (rem)
    {
      const u32 w = pw->i[pw_len >> 2];

      if (rem >= 1) update_key012(key0, key1, key2, unpack_v8a_from_v32(w));
      if (rem >= 2) update_key012(key0, key1, key2, unpack_v8b_from_v32(w));
      if (rem >= 3) update_key012(key0, key1, key2, unpack_v8c_from_v32(w));
    }

    u32 plain;
    u32 key3;
    u32 next;

    next = l_data[0];

    update_key3(key2, key3);
    plain = unpack_v8a_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8b_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8c_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8d_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    next = l_data[1];

    update_key3(key2, key3);
    plain = unpack_v8a_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8b_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8c_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8d_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    next = l_data[2];

    update_key3(key2, key3);
    plain = unpack_v8a_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8b_from_v32(next) ^ key3;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8c_from_v32(next) ^ key3;
    if ((checksum_size == 2) && ((checksum_from_crc & 0xff) != plain) && ((checksum_from_timestamp & 0xff) != plain)) return;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8d_from_v32(next) ^ key3;
    if ((plain != (checksum_from_crc >> 8)) && (plain != (checksum_from_timestamp >> 8))) return;
    update_key012(key0, key1, key2, plain);

    next = l_data[3];

    update_key3(key2, key3);
    plain = unpack_v8a_from_v32(next) ^ key3;
    const u32 btype = plain & 6;
    if (btype == 6) return;
    update_key012(key0, key1, key2, plain);

    if (data_length < 36)
    {
      match_out[gid] = 1;
      return;
    }

    // STRICT MODE: Only allow Fixed (2) or Dynamic (4) Huffman codes.
    if (btype != 2 && btype != 4) return;

    u32 tmp0 = plain << 0;
    u32 tmp1 = 0;
    u32 tmp2 = 0;
    u32 tmp3 = 0;
    u32 tmp4 = 0;
    u32 tmp5 = 0;

    update_key3(key2, key3);
    plain = unpack_v8b_from_v32(next) ^ key3;
    tmp0 |= plain << 8;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8c_from_v32(next) ^ key3;
    tmp0 |= plain << 16;
    update_key012(key0, key1, key2, plain);

    update_key3(key2, key3);
    plain = unpack_v8d_from_v32(next) ^ key3;
    tmp0 |= plain << 24;
    update_key012(key0, key1, key2, plain);

    next = l_data[4];

    update_key3(key2, key3); plain = unpack_v8a_from_v32(next) ^ key3; tmp1 |= plain << 0;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8b_from_v32(next) ^ key3; tmp1 |= plain << 8;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8c_from_v32(next) ^ key3; tmp1 |= plain << 16; update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8d_from_v32(next) ^ key3; tmp1 |= plain << 24; update_key012(key0, key1, key2, plain);

    next = l_data[5];

    update_key3(key2, key3); plain = unpack_v8a_from_v32(next) ^ key3; tmp2 |= plain << 0;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8b_from_v32(next) ^ key3; tmp2 |= plain << 8;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8c_from_v32(next) ^ key3; tmp2 |= plain << 16; update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8d_from_v32(next) ^ key3; tmp2 |= plain << 24; update_key012(key0, key1, key2, plain);

    next = l_data[6];

    update_key3(key2, key3); plain = unpack_v8a_from_v32(next) ^ key3; tmp3 |= plain << 0;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8b_from_v32(next) ^ key3; tmp3 |= plain << 8;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8c_from_v32(next) ^ key3; tmp3 |= plain << 16; update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8d_from_v32(next) ^ key3; tmp3 |= plain << 24; update_key012(key0, key1, key2, plain);

    next = l_data[7];

    update_key3(key2, key3); plain = unpack_v8a_from_v32(next) ^ key3; tmp4 |= plain << 0;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8b_from_v32(next) ^ key3; tmp4 |= plain << 8;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8c_from_v32(next) ^ key3; tmp4 |= plain << 16; update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8d_from_v32(next) ^ key3; tmp4 |= plain << 24; update_key012(key0, key1, key2, plain);

    next = l_data[8];

    update_key3(key2, key3); plain = unpack_v8a_from_v32(next) ^ key3; tmp5 |= plain << 0;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8b_from_v32(next) ^ key3; tmp5 |= plain << 8;  update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8c_from_v32(next) ^ key3; tmp5 |= plain << 16; update_key012(key0, key1, key2, plain);
    update_key3(key2, key3); plain = unpack_v8d_from_v32(next) ^ key3; tmp5 |= plain << 24; update_key012(key0, key1, key2, plain);

    if (btype == 2 && !check_inflate_code1(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, 24)) return;
    if (btype == 4 && !check_inflate_code2(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5))     return;

    // If we reached here, the password passed all early checks
    match_out[gid] = 1;
}
