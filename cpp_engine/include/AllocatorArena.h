#pragma once
#include <cstddef>

class ArenaAllocator {
private:
    char* buffer_inicio;
    size_t tamanho_total;
    size_t offset_atual;

public:
    ArenaAllocator(size_t tamanho_em_bytes); 
    ~ArenaAllocator();            
    
    void* alocar(size_t bytes_necessarios);    
    void resetar();

    // Acessores para o sistema de Ping-Pong (usados pela thread de drain)
    const char* getBuffer()    const { return buffer_inicio; }
    size_t      getUsedBytes() const { return offset_atual;  }
    size_t      getCapacity()  const { return tamanho_total;  }
};