#include "../include/AllocatorArena.h" // 1. Bring the header in here

// The Constructor
ArenaAllocator::ArenaAllocator(size_t tamanho_em_bytes) {
    tamanho_total = tamanho_em_bytes;
    offset_atual = 0; 
    buffer_inicio = new char[tamanho_total];
}

// The Destructor
ArenaAllocator::~ArenaAllocator() {
    delete [] buffer_inicio;
}

// The Allocate method (note the ::)
void* ArenaAllocator::alocar(size_t bytes_necessarios) {
    size_t limit = (tamanho_total - offset_atual);
    if (limit >= bytes_necessarios) {
        char* buffer_agora = offset_atual + buffer_inicio;
        offset_atual = offset_atual + bytes_necessarios;
        return buffer_agora;
    }
    else {
        return nullptr;
    }
}

// The Reset method
void ArenaAllocator::resetar() {
    offset_atual = 0;
}