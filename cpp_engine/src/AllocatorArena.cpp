#include "../include/AllocatorArena.h"

ArenaAllocator::ArenaAllocator(size_t size_bytes)
    : buffer_start(new char[size_bytes]), total_size(size_bytes), offset(0) {}

ArenaAllocator::~ArenaAllocator() {
    delete[] buffer_start;
}

void* ArenaAllocator::allocate(size_t bytes) {
    if (total_size - offset < bytes) {
        return nullptr;
    }
    char* p = buffer_start + offset;
    offset += bytes;
    return p;
}

void ArenaAllocator::reset() {
    offset = 0;
}
