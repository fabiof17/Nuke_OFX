// ChromaAberrations_OFX - minimal OpenEXR writer (scanline, uncompressed, 32-bit float)
// Copyright (C) 2026 ChromaAberrations_OFX authors. Dual-licensed: GPL v3 or commercial (see LICENSING.md).

#include "exrwriter.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace ca {

namespace {

struct Buf
{
    std::vector<unsigned char> b;
    void u8(unsigned v) { b.push_back( (unsigned char)v ); }
    void i32(int32_t v) { for (int i = 0; i < 4; ++i) b.push_back( (unsigned char)( ( (uint32_t)v >> (8 * i) ) & 0xff ) ); }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back( (unsigned char)( (v >> (8 * i)) & 0xff ) ); }
    void f32(float f) { uint32_t u; std::memcpy(&u, &f, 4); i32( (int32_t)u ); }
    void str(const std::string& s) { b.insert( b.end(), s.begin(), s.end() ); b.push_back(0); }
    void attr(const std::string& name, const std::string& type, const Buf& value)
    {
        str(name);
        str(type);
        i32( (int32_t)value.b.size() );
        b.insert( b.end(), value.b.begin(), value.b.end() );
    }
};

} // namespace

bool writeExr(const std::string& path, int width, int height, const std::vector<ExrChannel>& channelsIn, std::string* err)
{
    if ( (width <= 0) || (height <= 0) || channelsIn.empty() ) {
        if (err) *err = "empty image";
        return false;
    }
    // channels must be stored in alphabetical order
    std::vector<ExrChannel> ch = channelsIn;
    std::sort( ch.begin(), ch.end(), [](const ExrChannel& a, const ExrChannel& b) { return a.name < b.name; } );

    Buf h;
    h.i32(20000630);   // magic
    h.i32(2);          // version 2, single-part scanline
    {
        Buf v;
        for (const ExrChannel& c : ch) {
            v.str(c.name);
            v.i32(2);          // FLOAT
            v.u8(0); v.u8(0); v.u8(0); v.u8(0); // pLinear + reserved
            v.i32(1);
            v.i32(1);
        }
        v.u8(0);
        h.attr("channels", "chlist", v);
    }
    { Buf v; v.u8(0); h.attr("compression", "compression", v); }
    { Buf v; v.i32(0); v.i32(0); v.i32(width - 1); v.i32(height - 1); h.attr("dataWindow", "box2i", v); }
    { Buf v; v.i32(0); v.i32(0); v.i32(width - 1); v.i32(height - 1); h.attr("displayWindow", "box2i", v); }
    { Buf v; v.u8(0); h.attr("lineOrder", "lineOrder", v); }
    { Buf v; v.f32(1.f); h.attr("pixelAspectRatio", "float", v); }
    { Buf v; v.f32(0.f); v.f32(0.f); h.attr("screenWindowCenter", "v2f", v); }
    { Buf v; v.f32(1.f); h.attr("screenWindowWidth", "float", v); }
    h.u8(0); // end of header

    const uint64_t lineBytes = (uint64_t)width * 4 * ch.size();
    const uint64_t blockBytes = 8 + lineBytes;
    const uint64_t first = h.b.size() + 8ull * height;
    for (int y = 0; y < height; ++y) {
        h.u64( first + (uint64_t)y * blockBytes );
    }

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    bool ok = std::fwrite(h.b.data(), 1, h.b.size(), f) == h.b.size();
    std::vector<unsigned char> line( (size_t)blockBytes );
    for (int y = 0; y < height && ok; ++y) {
        unsigned char* p = line.data();
        const int32_t yy = y, sz = (int32_t)lineBytes;
        std::memcpy(p, &yy, 4);
        std::memcpy(p + 4, &sz, 4);
        p += 8;
        for (const ExrChannel& c : ch) {
            std::memcpy(p, c.data + (size_t)y * width, (size_t)width * 4);
            p += (size_t)width * 4;
        }
        ok = std::fwrite(line.data(), 1, line.size(), f) == line.size();
    }
    ok = (std::fclose(f) == 0) && ok;
    if (!ok && err) {
        *err = "write error on " + path;
    }
    return ok;
}

} // namespace ca
