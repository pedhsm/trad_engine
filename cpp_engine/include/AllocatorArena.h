#pragma once
#include <cstddef>

// Bump allocator over one fixed buffer: allocate() hands out the next `bytes`, reset()
// forgets everything at once. No per-allocation free, no system call after the
// constructor — which is what makes it usable on the market-data hot path.
class ArenaAllocator {
private:
    char* buffer_start;
    size_t total_size;
    size_t offset;

public:
    explicit ArenaAllocator(size_t size_bytes);
    ~ArenaAllocator();
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // nullptr when the arena has no room left (the caller decides what to drop).
    void* allocate(size_t bytes);
    void reset();

    // Accessors for the ping-pong drain thread.
    const char* getBuffer()    const { return buffer_start; }
    size_t      getUsedBytes() const { return offset; }
    size_t      getCapacity()  const { return total_size; }
};
