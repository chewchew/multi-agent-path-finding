#include "simulator.h"

#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

DEBUG_PLATFORM_FREE_FILE_MEMORY(DEBUG_win32_platform_free_file_memory)
{
    if (memory)
    {
        free(memory);
    }
}

DEBUG_PLATFORM_READ_ENTIRE_FILE(DEBUG_win32_platform_read_entire_file)
{
    DebugReadFileResult read_file_result = {};

    s32 file_handle = open(filename, O_RDONLY);
    Assert(file_handle != -1);
    off_t file_size = lseek(file_handle, 0, SEEK_END);
    void* contents = malloc(file_size);

    ssize_t bytes_read = pread(file_handle, contents, file_size, 0);
    Assert(bytes_read == file_size);
    
    read_file_result.contents = contents;
    read_file_result.contents_size = file_size;
    
    close(file_handle);
    
    Assert(read_file_result.contents);
    return read_file_result;
}

int main(int argc, char **argv)
{
    DEBUG_platform_free_file_memory_ = DEBUG_win32_platform_free_file_memory;
    DEBUG_platform_read_entire_file_ = DEBUG_win32_platform_read_entire_file;

    MemoryArena memory_arena;
    memory_arena.size = Megabytes(640);
    memory_arena.base = (u8*)mmap(0, memory_arena.size,
                                  PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE,
                                  -1, 0);
    memory_arena.used = 0;
    
    simulate(&memory_arena);

#if 0
    SimulatorState* state = (SimulatorState*)memory_arena.base;
    Graph* graph = state->graph;

    printf(" n:");
    for (u32 vertex_index = 0;
         vertex_index < graph->vertex_count;
         vertex_index++)
    {
        char str[255];
        sprintf(str, "%4d", vertex_index);
        printf(str);
    }
    printf("\n");
    for (u32 vertex_index = 0;
         vertex_index < graph->vertex_count;
         vertex_index++)
    {
        char str[255];
        sprintf(str, "%2d:", vertex_index);
        printf(str);
        for (u32 edge_index = 0;
             edge_index < graph->vertex_count;
             edge_index++)
        {
            s32 edge = (s32)graph->edges[vertex_index * graph->vertex_count + edge_index];
            if (edge == 0)
            {
                sprintf(str, "   .");
            }
            else
            {
                sprintf(str, "%4d", edge);
            }
            printf(str);
        }   
        printf("\n");
    }
#endif
}
