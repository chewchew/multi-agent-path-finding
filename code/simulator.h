#if SIMULATOR_DEBUG
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#endif

#include <stdint.h>

typedef int32_t b32;
typedef float f32;
typedef double f64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
    
#define global_variable static

#ifdef __linux__
#include <assert.h>
#define Assert(expression) assert(expression);
#else
#define Assert(expression) if(!(expression)) *(int*)0 = 0
#endif

#define ArrayCount(array) (sizeof(array) / sizeof((array)[0]))

#define Kilobytes(value) ((value) * 1024LL)
#define Megabytes(value) (Kilobytes(value) * 1024LL)

#define InvalidCodePath Assert(!"Invalid code path!")

#define IsCharacter(c) ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
#define IsDigit(c) (c >= '0' && c <= '9')
#define IsWhitespace(c) (c == ' ' || c == '\n' || c == '\r')

#define FloatEq(f1, f2) (fabs(f1 - f2) <= 0.001f)

static void
skip_whitespace(u32* i, char* str){  while(IsWhitespace(str[*i])) *i = *i + 1; if (IsWhitespace(str[*i])) *i = *i + 1; }

#if _MSC_VER
#define INIT_CLOCK(ID) u64 clock##ID = 0;
#define CLOCK_START(ID) u64 start##ID = __rdtsc();
#define CLOCK_END(ID) clock##ID += __rdtsc() - start##ID;
#endif

struct MemoryArena
{
    size_t size;
    u8* base;
    size_t used;
};

static void
initialize_arena(MemoryArena* arena, size_t size, u8* base)
{
    arena->size = size;
    arena->base = base;
    arena->used = 0;
}

#define zero_struct(struct_instance) zero_size(sizeof(struct_instance), &(struct_instance))
inline void
zero_size(size_t size_bytes, void* ptr)
{
    u8* byte = (u8*)ptr;
    while(size_bytes > 0)
    {
        *byte++ = 0;
        size_bytes--;
    }
}

#define push_size(arena, size)_push(arena, size)
#define push_array(arena, type, count)(type*)push_size(arena, sizeof(type) * count)
#define push_struct(arena, type) (type*)_push(arena, sizeof(type))
static void*
_push(MemoryArena* arena, size_t size, size_t alignment = 4)
{
    size_t result_pointer = (size_t)arena->base + arena->used;
    
    size_t alignment_offset = 0;
    
    size_t alignment_mask = alignment - 1;
    if (result_pointer & alignment_mask)
    {
        alignment_offset = alignment - result_pointer & alignment_mask;
    }
    size += alignment_offset;
    
    Assert(arena->used + size <= arena->size);
    arena->used += size;

    void* result = (void*)(result_pointer + alignment_offset);

    zero_size(size, result);
    
    return result;
}

inline u32
safe_truncate_size_64(u64 value)
{
    Assert(value <= 0xFFFFFFFF);
    u32 result = (u32)value;
    return result;
}

static u32
char_to_u32(char c)
{
    Assert(c >= '0' && c <= '9');
    return c - '0';
}

struct StringToU32Result
{
    u32 number;
    u32 length;
};

static StringToU32Result
string_to_u32(char* str, u32 from)
{
    StringToU32Result result;
    result.number = 0;
    
    u32 start = from;
    
    while (str[from] >= '0' && str[from] <= '9')
    {
        result.number *= 10;
        result.number += char_to_u32(str[from]);
        from++;
    }

    result.length = from - start;
    
    return result;
}

static u32
read_string(char* str, u32 from, char* buffer, u32 buffer_size)
{
    u32 string_length = 0;

    while (!IsWhitespace(str[from]))
    {
        buffer[string_length++] = str[from];
        from++;
    }
    
    return string_length;
}

static s32
contains_u32(u32* array, u32 size, u32 element)
{
    s32 result = -1;
    for (u32 i = 0; i < size; i++) if (array[i] == element) { result = i; break; }
    return result;
}

struct DebugReadFileResult
{
    u32 contents_size;
    void* contents;
};

#define DEBUG_PLATFORM_READ_ENTIRE_FILE(name) DebugReadFileResult name(const char* filename)
typedef DEBUG_PLATFORM_READ_ENTIRE_FILE(_debug_platform_read_entire_file);
DEBUG_PLATFORM_READ_ENTIRE_FILE(DEBUG_platform_read_entire_file_stub)
{
    DebugReadFileResult result = {};
    return result;
}
global_variable _debug_platform_read_entire_file* DEBUG_platform_read_entire_file_ = DEBUG_platform_read_entire_file_stub;
#define DEBUG_platform_read_entire_file DEBUG_platform_read_entire_file_

#define DEBUG_PLATFORM_FREE_FILE_MEMORY(name) void name(void* memory)
typedef DEBUG_PLATFORM_FREE_FILE_MEMORY(_debug_platform_free_file_memory);
DEBUG_PLATFORM_FREE_FILE_MEMORY(DEBUG_platform_free_file_memory_stub)
{
}
global_variable _debug_platform_free_file_memory* DEBUG_platform_free_file_memory_ = DEBUG_platform_free_file_memory_stub;
#define DEBUG_platform_free_file_memory DEBUG_platform_free_file_memory_

struct LinkedList
{
    u8* data;
    u32 data_size_bytes;
    LinkedList* next;

    void add(MemoryArena* memory_arena, u8* new_data, u32 new_data_size_bytes)
    {
        if (this->data == 0)
        {
            this->data = new_data;
            this->data_size_bytes = new_data_size_bytes;
        }
        else
        {
#if 0
            LinkedList* list = this;
            while (list->next)
            {
                list = list->next;
            }
            list->next = push_struct(memory_arena, LinkedList);
            list = list->next;
            list->data = new_data;
            list->data_size_bytes = new_data_size_bytes;
#else
            LinkedList* new_entry = push_struct(memory_arena, LinkedList);
            new_entry->next = this->next;
            new_entry->data = this->data;
            new_entry->data_size_bytes = this->data_size_bytes;
            this->next = new_entry;
            this->data = new_data;
            this->data_size_bytes = new_data_size_bytes;
#endif
        }
    }

    b32 empty()
    {
        if (this->data == 0) Assert(this->data_size_bytes == 0);
        return this->data == 0;
    }
};

struct Edge
{
    u32 to;
    f32 cost;
};

struct Vertex
{
    f32 x;
    f32 y;
    LinkedList* edges;
};

struct Graph
{
    u32 vertex_count;
    Vertex* vertices;
};

struct SIPPNode
{
    SIPPNode* parent;
    
    u32 vertex;
    f32 arrival_time;
    f32 safe_interval_end;
    f32 fscore;
};

struct WaitConflict
{
    u32 vertex;
};

struct MoveConflict
{
    u32 from;
    u32 to;
};

enum ActionType
{
    ACTION_TYPE_WAIT,
    ACTION_TYPE_MOVE
};

struct Interval
{
    f32 start;
    f32 end;
};

typedef Interval SafeInterval;
#define INF 10000000
#define DELTA 0

static b32
intersect(Interval i1, Interval i2)
{
    return ((i1.start <= i2.end && i1.end >= i2.start) ||
            (i2.start <= i1.end && i2.end >= i1.start));
}

static Interval
intersection(Interval i1, Interval i2)
{
    return {(f32)fmax(i1.start, i2.start), (f32)fmin(i1.end, i2.end)};
}

static b32
superset(Interval i1, Interval i2)
{
    return i1.start <= i2.start && i1.end >= i2.end;
}

static b32
interval_exists(Interval i, b32 inclusive = true)
{
    if (FloatEq(i.start, INF) && FloatEq(i.end, INF))
    {
        return false;
    }
    else if (inclusive)
    {
        return FloatEq(i.start, i.end) || i.start < i.end;
    }
    else
    {
        return i.start < i.end;
    }
}

static b32
in(f32 x, Interval i)
{
    return x >= i.start && x <= i.end;
}

struct Action
{
    Interval interval;
    
    ActionType type;
    union
    {
        u32 wait_vertex;
        struct
        {
            u32 from;
            u32 to;
        } move;
    };
};

struct Conflict
{
    Interval interval;
    
    u32 agent_1_id;
    Action action_1;
    
    u32 agent_2_id;
    Action action_2;
    
};

struct Constraint
{
    Interval interval;
    u32 agent_id;
    ActionType type;
    u32 from;
    u32 to;
};

static b32
constraints_equal(Constraint c1, Constraint c2)
{
    b32 result = false;
    
    if (FloatEq(c1.interval.start, c2.interval.start) &&
        FloatEq(c1.interval.end, c2.interval.end) &&
        c1.agent_id == c2.agent_id &&
        c1.type == c2.type)
    {
        if (c1.type == ACTION_TYPE_MOVE)
        {
            return c1.from == c2.from && c1.to == c2.to;
        }
        else if (c1.type == ACTION_TYPE_WAIT)
        {
            return c1.from == c2.from;
        }
        else InvalidCodePath;
    }

    return result;
}

struct CBSNode
{
    CBSNode* parent;
    Constraint constraint;
    f32 cost;
};

#if 0
static b32
cbs_node_equal(MemoryArena* memory_arena, CBSNode node1, CBSNode node2)
{
    size_t tmp = memory_arena->used;
    
    b32 result = true;

    u32 constraint_count1 = 0;
    Conflicts* p = node1.conflicts;
    while (p) { constraint_count1++; p = p->next; }
    Constraint* constraints1 = push_array(memory_arena, Constraint, constraint_count1 + 1);
    p = node1.conflicts;
    u32 i = 1;
    constraints1[0] ={ node1.new_conflict.interval,
                       node1.new_conflict.agent_1_id,
                       node1.new_conflict.action_1.type,
                       node1.new_conflict.action_1.move.from,
                       node1.new_conflict.action_1.move.to };
    while (p)
    {
        constraints1[i++] = { p->interval,
                              p->agent_1_id,
                              p->action_1.type,
                              p->action_1.move.from,
                              p->action_1.move.to };
        p = p->next;
    }

    u32 constraint_count2 = 0;
    p = node2.conflicts;
    while (p) { constraint_count2++; p = p->next; }
    Constraint* constraints2 = push_array(memory_arena, Constraint, constraint_count2 + 1);
    p = node2.conflicts;
    i = 1;
    constraints2[0] ={ node2.new_conflict.interval,
                       node2.new_conflict.agent_1_id,
                       node2.new_conflict.action_1.type,
                       node2.new_conflict.action_1.move.from,
                       node2.new_conflict.action_1.move.to };
    while (p)
    {
        constraints2[i++] = { p->interval,
                              p->agent_1_id,
                              p->action_1.type,
                              p->action_1.move.from,
                              p->action_1.move.to };
        p = p->next;
    }

    b32* checked = push_array(memory_arena, b32, constraint_count2);
    
    if (constraint_count1 == constraint_count2)
    {
        for (i = 0; i < constraint_count1; i++)
        {
            for (u32 j = 0; j < constraint_count2; j++)
            {
                if (!checked[j] &&
                    constraints_equal(constraints1[i], constraints2[j]))
                {
                    checked[j] = true;
                    break;
                }
            }
            
        }

        for (i = 0; i < constraint_count1; i++)
        {
            if (!checked[i])
            {
                result = false;
                break;
            }
        }
    }
    else
    {
        result = false;
    }

    memory_arena->used = tmp;
    
    return result;
}
#endif

struct Path
{
    u32 vertex_count;
    SIPPNode* vertices;
};

struct SIPPQueue
{
    /* SIPPQueue* prev; */
    SIPPQueue* next;
    SIPPNode* node;
    b32 is_used;
};

struct CBSQueue
{
    CBSQueue* next;
    CBSNode* node;
    b32 is_used;
};

#if 0
static b32
cbs_node_in_array(MemoryArena* memory_arena, CBSNode node, CBSNode* array, u32 count)
{
    b32 result = false;
    
    for (u32 i = 0; i < count; i++)
    {
        if (cbs_node_equal(memory_arena, node, array[i]))
        {
            result = true;
            break;
        }
    }

    return result;
}
#endif

static SIPPQueue*
add_astar_node(MemoryArena* memory_arena, SIPPQueue* queue, SIPPNode* node)
{
    if (!queue)
    {
        queue = push_struct(memory_arena, SIPPQueue);
    }

    if (!queue->is_used)
    {
        queue->node = node;
        queue->is_used = true;
    }
    else
    {
        SIPPQueue* next = push_struct(memory_arena, SIPPQueue);
        next->node = node;
        next->is_used = true;

        SIPPQueue* iter_prev = queue;
        SIPPQueue* iter = queue;
        while (iter && node->fscore >= iter->node->fscore)
        {
            iter_prev = iter;
            iter = iter->next;
        }

        if (iter)
        {
            if (iter == queue)
            {
                next->next = queue;
                queue = next;
            }
            else
            {
                SIPPQueue* tmp = iter_prev->next;
                iter_prev->next = next;
                next->next = tmp;
            }
        }
        else
        {
            iter_prev->next = next;
        }
    }


#if 0
    else if(!queue->next)
    {
        SIPPQueue* next = push_struct(memory_arena, SIPPQueue);
        next->node = node;
        next->is_used = true;

        if (node->fscore < queue->node->fscore)
        {
            next->prev = 0;
            next->next = queue;
            queue->prev = next;
            queue = next;
        }
        else
        {
            queue->next = next;
            next->prev = queue;
        }
    }
    else if(node->fscore < queue->node->fscore)
    {
        SIPPQueue* next = push_struct(memory_arena, SIPPQueue);
        next->node = node;
        next->is_used = true;
        
        next->prev = 0;
        next->next = queue;
        queue->prev = next;
        queue = next;
    }
    else
    {
        SIPPQueue* next = push_struct(memory_arena, SIPPQueue);
        next->node = node;
        next->is_used = true;

        SIPPQueue* iter = queue;
        while (iter->next &&
               node->fscore >= iter->node->fscore)
        {
            iter = iter->next;
        }

        SIPPQueue* iter_prev = iter->prev;
        if (iter_prev)
        {
            iter_prev->next = next;
            next->prev = iter_prev;
        }
        
        iter->prev = next;
        next->next = iter;
    }
#endif
    
    return queue;
}

static SIPPNode*
create_sipp_node(MemoryArena* memory_arena,
                 SIPPNode* parent, u32 vertex,
                 f32 earliest_arrival_time, f32 safe_interval_end,
                 f32 move_time, f32 hvalue)
{
    SIPPNode* new_node = push_struct(memory_arena, SIPPNode);
    new_node->parent = parent;
    new_node->vertex = vertex;
    new_node->arrival_time = earliest_arrival_time;
    new_node->safe_interval_end = safe_interval_end;
    new_node->fscore = earliest_arrival_time + hvalue;
    return new_node;
}

struct TimeNode
{
    TimeNode* next;
    f32 time;
};

static b32
has_visited(TimeNode* node, f32 time)
{
    b32 result = false;
    f32 delta = 0.001f;
    while (node)
    {
        if (FloatEq(time, node->time))
        {
            result = true;
            break;
        }
        node = node->next;
    }
    return result;
}

static b32
can_wait_forever(SafeInterval* safe_intervals, u32 safe_interval_count, f32 from_time)
{
    b32 result = false;
    if (safe_interval_count == 0)
    {
        result = true;
    }
    else
    {
        for (u32 interval_index = 0;
             interval_index < safe_interval_count;
             interval_index++)
        {
            SafeInterval interval = safe_intervals[interval_index];
            if (interval.start <= from_time && FloatEq(interval.end, INF))
            {
                result = true;
                break;
            }
        }
    }
    return result;
}

struct SafeIntervalCold
{
    SafeIntervalCold* prev;
    SafeIntervalCold* next;
    SafeInterval interval;
};

struct ComputeSafeIntervalsResult
{
    SafeInterval** safe_intervals_vertex;
    u32* safe_intervals_vertex_count;
    SafeInterval** safe_intervals_edge;
    u32* safe_intervals_edge_count;
};

static SafeIntervalCold*
remove_interval(MemoryArena* memory_arena, SafeIntervalCold* intervals,
                Interval conflict_interval, u32* safe_intervals_count, u32 interval_index)
{
    SafeIntervalCold* current = intervals;
    
    while (current)
    {
        SafeInterval safe_interval = current->interval;

        SafeInterval split_interval = intersection(conflict_interval, safe_interval);
        if (interval_exists(split_interval, false))
        {
            SafeInterval lower_interval = {safe_interval.start, split_interval.start - DELTA};
            SafeInterval upper_interval = {split_interval.end + DELTA, safe_interval.end};

            if (!interval_exists(lower_interval, false))
            {
                if (!interval_exists(upper_interval, false))
                {
                    safe_intervals_count[interval_index]--;
                    if (current->next)
                    {
                        current->next->prev = current->prev;
                    }

                    if (current->prev)
                    {
                        current->prev->next = current->next;
                    }
                    else
                    {
                        intervals = intervals->next;
                    }
                }
                else
                {
                    current->interval = upper_interval;
                }
            }
            else if (!interval_exists(upper_interval, false))
            {
                current->interval = lower_interval;
            }
            else
            {
                safe_intervals_count[interval_index]++;

                current->interval = lower_interval;
                            
                SafeIntervalCold* new_interval = push_struct(memory_arena, SafeIntervalCold);
                new_interval->interval = upper_interval;

                SafeIntervalCold* tmp = current->next;
                current->next = new_interval;
                new_interval->next = tmp;
                new_interval->prev = current;
                if (tmp)
                {
                    tmp->prev = new_interval;
                }
                            
                current = new_interval;
            }
        }

        current = current->next;
    };
    
    return intervals;
}

static u32
get_edge_index(u32 vertex_count, u32 from, u32 to)
{
    return vertex_count * from + to;
}

static Path
sipp(MemoryArena* memory_arena,
     Graph* graph,
     u32 start, u32 goal,
     f32 (*heuristic)(Graph*, u32, u32),
     ComputeSafeIntervalsResult safe_intervals)
{
    SafeInterval** safe_intervals_vertex = safe_intervals.safe_intervals_vertex;
    u32* safe_intervals_vertex_count = safe_intervals.safe_intervals_vertex_count;
    SafeInterval** safe_intervals_edge = safe_intervals.safe_intervals_edge;
    u32* safe_intervals_edge_count = safe_intervals.safe_intervals_edge_count;
    
    Path result = {};

    u32 queue_count = 0;
    SIPPQueue* queue = 0;
    SIPPNode* root = push_struct(memory_arena, SIPPNode);
    root->vertex = start;
    root->arrival_time = 0;
    if (safe_intervals_vertex[start][0].start > 0)
    {
        root->safe_interval_end = 0;
    }
    else
    {
        root->safe_interval_end = safe_intervals_vertex[start][0].end;
    }
    root->fscore = heuristic(graph, start, goal);

    queue = add_astar_node(memory_arena, queue, root);
    queue_count++;

    TimeNode** visited = push_array(memory_arena, TimeNode*, graph->vertex_count);

    u32 expansions = 0;
    while(queue_count > 0)
    {
#if 0
        {
            size_t tmp_used = memory_arena->used;
            u32 path_length = 0;
            SIPPNode* p = queue->node;
            while (p) { path_length++; p = p->parent; }
            SIPPNode* path = push_array(memory_arena, SIPPNode, path_length);
            p = queue->node;
            u32 i = 0;
            while (p) { path[i++] = *p; p = p->parent; }
            memory_arena->used = tmp_used;
        }

        {
            size_t tmp_used = memory_arena->used;
            SIPPQueue* p = queue;
            SIPPNode* _queue = push_array(memory_arena, SIPPNode, queue_count);
            u32 i = 0;
            while (p) { _queue[i++] = *p->node; p = p->next; }
            memory_arena->used = tmp_used;
        }
#endif
        
        expansions++;
        SIPPNode* current_node = queue->node;
        queue = queue->next;
        queue_count--;
        
        u32 current_vertex = current_node->vertex;
        f32 arrival_time = current_node->arrival_time;
        f32 safe_interval_end = current_node->safe_interval_end;
        
        Assert(current_vertex < graph->vertex_count);
        
        if (current_vertex == goal && can_wait_forever(safe_intervals_vertex[current_vertex], safe_intervals_vertex_count[current_vertex], arrival_time))
        {
            // NOTE: Reconstruct path from best_path_vertex
            u32 path_length = 0;
            SIPPNode* prev = current_node;
            SIPPNode* next = current_node;
            SIPPNode path[255];
            path[path_length++] = *next;
            while (next->parent)
            {
                Assert(path_length < 255);
                next = next->parent;
                Assert(prev->arrival_time > next->arrival_time);
                path[path_length++] = *next;
                prev = next;
            }
            
            result.vertex_count = path_length;
            result.vertices = push_array(memory_arena, SIPPNode, result.vertex_count);//(SIPPNode*)malloc(sizeof(SIPPNode) * result.vertex_count);
            next = current_node;
            for (u32 vertex_index = 0;
                 vertex_index < path_length;
                 vertex_index++)
            {
                result.vertices[vertex_index] = path[result.vertex_count - 1 - vertex_index];
            }

#if 0 
            // TODO: fix root departure_time
            u32 neighbour_id = result.vertices[1].vertex;
            LinkedList* neighbour = graph->vertices[start].edges;
            b32 found_neighbour = false;
            f32 neighbour_cost = 0;
            while (neighbour && !found_neighbour)
            {
                Edge* edge = (Edge*)neighbour->data;
                if (edge->to == neighbour_id)
                {
                    neighbour_cost = edge->cost;
                    found_neighbour = true;
                    break;
                }
                neighbour = neighbour->next;
            }
            Assert(found_neighbour);
            result.vertices[0].departure_time = result.vertices[1].arrival_time - neighbour_cost;
#endif
            
            break;
        }

        LinkedList* current_neighbour = graph->vertices[current_vertex].edges;
        while (current_neighbour && !current_neighbour->empty())
        {
            Edge* edge = (Edge*)current_neighbour->data;
            f32 neighbour_cost = edge->cost;
            u32 neighbour_to = edge->to;
            f32 hvalue = heuristic(graph, neighbour_to, goal);
            current_neighbour = current_neighbour->next;

            f32 earliest_departure_time = arrival_time;
            f32 earliest_arrival_time = earliest_departure_time + neighbour_cost;
            f32 latest_departure_time = safe_interval_end;
            f32 latest_arrival_time = latest_departure_time + neighbour_cost;
            SafeInterval arrival_interval = {earliest_arrival_time, latest_arrival_time};

            for (u32 vertex_interval_index = 0;
                 vertex_interval_index < safe_intervals_vertex_count[neighbour_to];
                 vertex_interval_index++)
            {
                SafeInterval vertex_interval = safe_intervals_vertex[neighbour_to][vertex_interval_index];
                SafeInterval safe_arrival_interval_vertex = intersection(arrival_interval, vertex_interval);
                if (interval_exists(safe_arrival_interval_vertex))
                {
                    SafeInterval safe_departure_interval_vertex = {safe_arrival_interval_vertex.start - neighbour_cost,
                                                                   safe_arrival_interval_vertex.end - neighbour_cost};
                    u32 edge_index = get_edge_index(graph->vertex_count, current_vertex,  neighbour_to);
                    for (u32 edge_interval_index = 0;
                         edge_interval_index < safe_intervals_edge_count[edge_index];
                         edge_interval_index++)
                    {
                        SafeInterval edge_interval = safe_intervals_edge[edge_index][edge_interval_index];
                            
                        SafeInterval safe_arrival_interval_edge = intersection(safe_arrival_interval_vertex, edge_interval);
                        SafeInterval safe_departure_interval_edge = intersection(safe_departure_interval_vertex, edge_interval);

                        if (!interval_exists(safe_arrival_interval_edge) ||
                            !interval_exists(safe_departure_interval_edge))
                        {
                            continue;
                        }
                        else if (in(safe_departure_interval_edge.start + neighbour_cost, safe_arrival_interval_edge))
                        {
                            earliest_arrival_time = safe_departure_interval_edge.start + neighbour_cost;
                        }
                        else if (in(safe_arrival_interval_edge.start - neighbour_cost, safe_departure_interval_edge))
                        {
                            earliest_arrival_time = safe_arrival_interval_edge.start - neighbour_cost;
                        }
                        else
                        {
                            continue;
                        }

                        if (!has_visited(visited[neighbour_to], earliest_arrival_time))
                        {
                            SIPPNode* new_node = create_sipp_node(memory_arena, current_node, neighbour_to,
                                                                  earliest_arrival_time, vertex_interval.end,
                                                                  neighbour_cost, hvalue);
                            queue = add_astar_node(memory_arena, queue, new_node);
                            queue_count++;

                            
                            TimeNode* time_node = push_struct(memory_arena, TimeNode);
                            time_node->next = visited[neighbour_to];
                            time_node->time = earliest_arrival_time;
                            visited[neighbour_to] = time_node;
                        }
                    }
                }
            }
        }
    }

    return result;
}

struct AgentInfo
{
    u32 start;
    u32 goal;
};

struct Solution
{
    Path* paths;
    u32 path_count;
};

static ComputeSafeIntervalsResult
compute_safe_intervals(MemoryArena* memory_arena, Graph* graph, Constraint* constraints, u32 constraint_count)
{
    ComputeSafeIntervalsResult result;
    
    u32 safe_intervals_vertex_size = graph->vertex_count;
    SafeIntervalCold** safe_intervals_vertex_cold = push_array(memory_arena, SafeIntervalCold*, safe_intervals_vertex_size);
    u32* safe_intervals_vertex_count = push_array(memory_arena, u32, safe_intervals_vertex_size);
    for (u32 vertex_index = 0;
         vertex_index < safe_intervals_vertex_size;
         vertex_index++)
    {
        safe_intervals_vertex_cold[vertex_index] = push_struct(memory_arena, SafeIntervalCold);
        safe_intervals_vertex_cold[vertex_index]->interval.start = 0;
        safe_intervals_vertex_cold[vertex_index]->interval.end = INF;
        safe_intervals_vertex_count[vertex_index] = 1;
    }

    u32 safe_intervals_edge_size = graph->vertex_count * graph->vertex_count;
    SafeIntervalCold** safe_intervals_edge_cold = push_array(memory_arena, SafeIntervalCold*, safe_intervals_edge_size);
    u32* safe_intervals_edge_count = push_array(memory_arena, u32, safe_intervals_edge_size);
    for (u32 edge_index = 0;
         edge_index < safe_intervals_edge_size;
         edge_index++)
    {
        safe_intervals_edge_cold[edge_index] = push_struct(memory_arena, SafeIntervalCold);
        safe_intervals_edge_cold[edge_index]->interval.start = 0;
        safe_intervals_edge_cold[edge_index]->interval.end = INF;
        safe_intervals_edge_count[edge_index] = 1;
    }

    for (u32 constraint_index = 0;
         constraint_index < constraint_count;
         constraint_index++)
    {
        Constraint constraint = constraints[constraint_index];

        // TODO: ta bort kollisions-intervallet från safe intervals
        if (constraint.type == ACTION_TYPE_WAIT)
        {
            u32 vertex_index = constraint.from;
            Interval interval = constraint.interval;
                
            SafeIntervalCold* iter2 = safe_intervals_vertex_cold[vertex_index];
            safe_intervals_vertex_cold[vertex_index] = remove_interval(memory_arena, iter2, interval, safe_intervals_vertex_count, vertex_index);
        }
        else if (constraint.type == ACTION_TYPE_MOVE)
        {
            u32 from_vertex_index = constraint.from;
            u32 to_vertex_index = constraint.to;
            Interval interval = constraint.interval;

            u32 edge_index = get_edge_index(graph->vertex_count, from_vertex_index, to_vertex_index);
                
            SafeIntervalCold* iter2 = safe_intervals_edge_cold[edge_index];
            safe_intervals_edge_cold[edge_index] = remove_interval(memory_arena, iter2, interval, safe_intervals_edge_count, edge_index);
        }
    };

    SafeInterval** safe_intervals_vertex = push_array(memory_arena, SafeInterval*, safe_intervals_vertex_size);
    for (u32 vertex_index = 0;
         vertex_index < safe_intervals_vertex_size;
         vertex_index++)
    {
        u32 safe_interval_count = safe_intervals_vertex_count[vertex_index];
        if (safe_interval_count > 0)
        {
            safe_intervals_vertex[vertex_index] = push_array(memory_arena, SafeInterval, safe_interval_count);
            SafeIntervalCold* iter = safe_intervals_vertex_cold[vertex_index];
            u32 safe_interval_index = 0;
            while (iter)
            {
                if (interval_exists(iter->interval))
                {
                    safe_intervals_vertex[vertex_index][safe_interval_index++] = iter->interval;
                }
                iter = iter->next;
            }
        }
    }
    
    SafeInterval** safe_intervals_edge = push_array(memory_arena, SafeInterval*, safe_intervals_edge_size);
    for (u32 edge_index = 0;
         edge_index < safe_intervals_edge_size;
         edge_index++)
    {
        u32 safe_interval_count = safe_intervals_edge_count[edge_index];
        if (safe_interval_count > 0)
        {
            safe_intervals_edge[edge_index] = push_array(memory_arena, SafeInterval, safe_interval_count);
            SafeIntervalCold* iter = safe_intervals_edge_cold[edge_index];
            u32 safe_interval_index = 0;
            while (iter)
            {
                if (interval_exists(iter->interval))
                {
                    safe_intervals_edge[edge_index][safe_interval_index++] = iter->interval;
                }
                iter = iter->next;
            }
        }
    }

    result.safe_intervals_vertex = safe_intervals_vertex;
    result.safe_intervals_vertex_count = safe_intervals_vertex_count;
    result.safe_intervals_edge = safe_intervals_edge;
    result.safe_intervals_edge_count = safe_intervals_edge_count;
    
    return result;
}

f32 h(Graph* g, u32 vertex, u32 goal)
{
    f32 x = g->vertices[vertex].x - g->vertices[goal].x;
    f32 y = g->vertices[vertex].y - g->vertices[goal].y;
    
    return sqrtf(x * x + y * y);
}        

static f32
cost_path(Path path)
{
    return path.vertices[path.vertex_count - 1].arrival_time;
}

static f32
cost(Path* paths, u32 agent_count)
{
    f32 result = 0;

    for (u32 path_index = 0;
         path_index < agent_count;
         path_index++)
    {
        Path path = paths[path_index];
        result += cost_path(paths[path_index]);
    }
    
    return result;
}

static CBSQueue*
add_cbs_node(MemoryArena* memory_arena, CBSQueue* queue, CBSNode* node)
{
    if (!queue)
    {
        queue = push_struct(memory_arena, CBSQueue);
    }

    if (!queue->is_used)
    {
        queue->node = node;
        queue->is_used = true;
    }
    else
    {
        CBSQueue* next = push_struct(memory_arena, CBSQueue);
        next->node = node;
        next->is_used = true;

        CBSQueue* iter_prev = queue;
        CBSQueue* iter = queue;
        while (iter && node->cost >= iter->node->cost)
        {
            iter_prev = iter;
            iter = iter->next;
        }

        if (iter)
        {
            if (iter == queue)
            {
                next->next = queue;
                queue = next;
            }
            else
            {
                CBSQueue* tmp = iter_prev->next;
                iter_prev->next = next;
                next->next = tmp;
            }
        }
        else
        {
            iter_prev->next = next;
        }
#if 0
        CBSQueue* next = push_struct(memory_arena, CBSQueue);
        next->node = node;
        next->is_used = true;
        CBSQueue* iter = queue;
        while (iter->next &&
               node.cost > iter->node.cost)
        {
            iter = iter->next;
        }
        CBSQueue* iter_next = iter->next;
        iter->next = next;
        next->next = iter_next;
#endif
    }

    return queue;
}

struct ActionIntervals
{
    Interval wait;
    Interval move;
};

static ActionIntervals
create_action_intervals(SIPPNode vertex, b32 last_vertex,
                        SIPPNode next_vertex, Graph* graph)
{
    ActionIntervals result;
    
    f32 arrival_time_vertex = vertex.arrival_time;
/* TODO: hämta cost ifrån grafen och använd istället för -1 */
/* graph->vertices[vertex.vertex]; */
    f32 departure_time_vertex;
    f32 leave_time_vertex;
    if (last_vertex)
    {
        departure_time_vertex = vertex.safe_interval_end;
        leave_time_vertex = vertex.safe_interval_end;
    }
    else
    {
        departure_time_vertex = next_vertex.arrival_time - 1;
        leave_time_vertex = next_vertex.arrival_time;
    }
    result.wait = {arrival_time_vertex, departure_time_vertex};
    result.move = {departure_time_vertex, leave_time_vertex};

    return result;
}

struct FindConflictResult
{
    b32 no_conflict_found;
    Conflict conflict;
};

static FindConflictResult
find_conflict(Path* path_buffer, u32 agent_count, Graph* graph)
{
    FindConflictResult result = {};
    result.no_conflict_found = true;
    
    for (u32 agent_index = 0;
         agent_index < agent_count && result.no_conflict_found;
         agent_index++)
    {
        Path path = path_buffer[agent_index];
        for (u32 vertex_index = 0;
             vertex_index < path.vertex_count && result.no_conflict_found;
             vertex_index++)
        {
            SIPPNode vertex = path.vertices[vertex_index];
            SIPPNode next_vertex = path.vertices[(vertex_index + 1) % path.vertex_count];
            b32 last_vertex = vertex_index == path.vertex_count - 1;
            ActionIntervals action_intervals_vertex = create_action_intervals(vertex, last_vertex, next_vertex, graph);
            
            for (u32 other_agent_index = agent_index + 1;
                 other_agent_index < agent_count && result.no_conflict_found;
                 other_agent_index++)
            {
                Path other_path = path_buffer[other_agent_index];
                for (u32 other_vertex_index = 0;
                     other_vertex_index < other_path.vertex_count && result.no_conflict_found;
                     other_vertex_index++)
                {
                    SIPPNode other_vertex = other_path.vertices[other_vertex_index];
                    SIPPNode next_other_vertex = other_path.vertices[(other_vertex_index + 1) % other_path.vertex_count];
                    b32 last_other_vertex = other_vertex_index == other_path.vertex_count - 1;
                    ActionIntervals action_intervals_other_vertex = create_action_intervals(other_vertex, last_other_vertex, next_other_vertex, graph);

                    Interval move_move_interval = intersection(action_intervals_vertex.move, action_intervals_other_vertex.move);
                    b32 move_move_interval_ok = move_move_interval.start != move_move_interval.end && interval_exists(move_move_interval);

                    Interval move_wait_interval = intersection(action_intervals_vertex.move, action_intervals_other_vertex.wait);
                    b32 move_wait_interval_ok = move_wait_interval.start != move_wait_interval.end && interval_exists(move_wait_interval);

                    Interval wait_move_interval = intersection(action_intervals_vertex.wait, action_intervals_other_vertex.move);
                    b32 wait_move_interval_ok = wait_move_interval.start != wait_move_interval.end && interval_exists(wait_move_interval);
                    
                    Interval wait_wait_interval = intersection(action_intervals_vertex.wait, action_intervals_other_vertex.wait);
                    b32 wait_wait_interval_ok = wait_wait_interval.start != wait_wait_interval.end && interval_exists(wait_wait_interval);
                                        
                    // NOTE: Conflict where both move into same vertex
                    if (!last_vertex && !last_other_vertex &&
                        next_vertex.vertex == next_other_vertex.vertex &&
                        move_move_interval_ok)
                    {
                        result.no_conflict_found = false;
                        result.conflict.interval = move_move_interval;

                        result.conflict.agent_1_id = agent_index;
                        result.conflict.action_1 = {action_intervals_vertex.move, ACTION_TYPE_MOVE};
                        result.conflict.action_1.move.from = vertex.vertex;
                        result.conflict.action_1.move.to = next_vertex.vertex;

                        result.conflict.agent_2_id = other_agent_index;
                        result.conflict.action_2 = {action_intervals_other_vertex.move, ACTION_TYPE_MOVE};
                        result.conflict.action_2.move.from = other_vertex.vertex;
                        result.conflict.action_2.move.to = next_other_vertex.vertex;
                    }
                    // NOTE: Conflict where both collide on edge
                    else if (!last_vertex && !last_other_vertex &&
                             next_vertex.vertex == other_vertex.vertex &&
                             vertex.vertex == next_other_vertex.vertex &&
                             move_move_interval_ok)
                    {
                        result.no_conflict_found = false;
                        result.conflict.interval = move_move_interval;

                        result.conflict.agent_1_id = agent_index;
                        result.conflict.action_1 = {action_intervals_vertex.move, ACTION_TYPE_MOVE};
                        result.conflict.action_1.move.from = vertex.vertex;
                        result.conflict.action_1.move.to = next_vertex.vertex;

                        result.conflict.agent_2_id = other_agent_index;
                        result.conflict.action_2 = {action_intervals_other_vertex.move, ACTION_TYPE_MOVE};
                        result.conflict.action_2.move.from = other_vertex.vertex;
                        result.conflict.action_2.move.to = next_other_vertex.vertex;
                    }
                    // NOTE: Conflict where prime agent move into waiting agent
                    else if (!last_vertex &&
                             next_vertex.vertex == other_vertex.vertex &&
                             move_wait_interval_ok)
                    {
                        result.no_conflict_found = false;
                        result.conflict.interval = move_wait_interval;

                        result.conflict.agent_1_id = agent_index;
                        result.conflict.action_1 = {action_intervals_vertex.move, ACTION_TYPE_MOVE};
                        result.conflict.action_1.move.from = vertex.vertex;
                        result.conflict.action_1.move.to = next_vertex.vertex;

                        result.conflict.agent_2_id = other_agent_index;
                        result.conflict.action_2 = {action_intervals_other_vertex.wait, ACTION_TYPE_WAIT, other_vertex.vertex};
                    }
                    else if (!last_other_vertex &&
                             vertex.vertex == next_other_vertex.vertex &&
                             wait_move_interval_ok)
                    {
                        result.no_conflict_found = false;
                        result.conflict.interval = wait_move_interval;

                        result.conflict.agent_1_id = agent_index;
                        result.conflict.action_1 = {action_intervals_vertex.wait, ACTION_TYPE_WAIT, vertex.vertex};

                        result.conflict.agent_2_id = other_agent_index;
                        result.conflict.action_2 = {action_intervals_other_vertex.move, ACTION_TYPE_MOVE};
                        result.conflict.action_2.move.from = other_vertex.vertex;
                        result.conflict.action_2.move.to = next_other_vertex.vertex;
                    }
                    else if (vertex.vertex == other_vertex.vertex && wait_wait_interval_ok)
                    {
                        result.no_conflict_found = false;
                        result.conflict.interval = wait_wait_interval;;

                        result.conflict.agent_1_id = agent_index;
                        result.conflict.action_1 = {action_intervals_vertex.wait, ACTION_TYPE_WAIT, vertex.vertex};

                        result.conflict.agent_2_id = other_agent_index;
                        result.conflict.action_2 = {action_intervals_other_vertex.wait, ACTION_TYPE_WAIT, other_vertex.vertex};
                    }
                }
            }
        }
    }
    
    return result;
}

struct MakeConstraintResult
{
    Constraint* data;
    u32 count;
};
    
static MakeConstraintResult
make_constraints(MemoryArena* memory_arena, CBSNode* node, u32 agent_id)
{
    MakeConstraintResult result;

    u32 constraint_count = 0;
    CBSNode* p = node;
    while (p->parent)
    {
        if (p->constraint.agent_id == agent_id)
        {
            constraint_count++;
        }
        p = p->parent;
    }
    Constraint* constraints = push_array(memory_arena, Constraint, constraint_count);

    p = node;
    u32 constraint_index = 0;
    while (p->parent)
    {
        if (p->constraint.agent_id == agent_id)
        {
            constraints[constraint_index++] = p->constraint;
        }
        p = p->parent;
    }

    result.data = constraints;
    result.count = constraint_count;

    return result;
}

#if SIMULATOR_DEBUG
global_variable u32 max_visited = 100000;
global_variable u32 visited_count_1 = 0;
global_variable ComputeSafeIntervalsResult* visited_1;
global_variable u32 visited_count_2 = 0;
global_variable ComputeSafeIntervalsResult* visited_2;
#endif

static CBSNode*
create_cbs_node(MemoryArena* memory_arena, CBSNode* node,
                Constraint constraint, Graph* graph, AgentInfo* agents,
                Path* path_buffer, u32 agent_count, u32 agent_id)
{
    CBSNode* result = 0;
    
    b32 node_1_ok = true;
    f32 cost_1 = 0;
    CBSNode* node_1_tmp = push_struct(memory_arena, CBSNode);
    node_1_tmp->parent = node;
    node_1_tmp->constraint = constraint;
    MakeConstraintResult constraints = make_constraints(memory_arena, node_1_tmp, agent_id);
    ComputeSafeIntervalsResult safe_intervals = compute_safe_intervals(memory_arena, graph, constraints.data, constraints.count);
#if SIMULATOR_DEBUG
    u32 count = agent_id == 0 ? visited_count_1 : visited_count_2;
    for (u32 i = 0; i < count && constraints.count > 0; i++)
    {
        b32 eq = true;
        ComputeSafeIntervalsResult x;
        if (agent_id == 0) x = visited_1[i];
        else x = visited_2[i];
        
        for (u32 j = 0; j < graph->vertex_count; j++)
        {
            for (u32 k = 0; k < x.safe_intervals_vertex_count[j]; k++)
            {
                if (x.safe_intervals_vertex_count[j] !=
                    safe_intervals.safe_intervals_vertex_count[j])
                {
                    eq = false;
                    break;
                }
                        
                if (!FloatEq(x.safe_intervals_vertex[j][k].start,
                             safe_intervals.safe_intervals_vertex[j][k].start) ||
                    !FloatEq(x.safe_intervals_vertex[j][k].end,
                             safe_intervals.safe_intervals_vertex[j][k].end))
                {
                    eq = false;
                    break;
                }
            }

            for (u32 jj = 0; jj < graph->vertex_count; jj++)
            {
                u32 edge_index = j * graph->vertex_count + jj;
                for (u32 k = 0; k < x.safe_intervals_edge_count[edge_index]; k++)
                {
                    if (x.safe_intervals_edge_count[edge_index] !=
                        safe_intervals.safe_intervals_edge_count[edge_index])
                    {
                        eq = false;
                        break;
                    }
                        
                    if (!FloatEq(x.safe_intervals_edge[edge_index][k].start,
                                 safe_intervals.safe_intervals_edge[edge_index][k].start) ||
                        !FloatEq(x.safe_intervals_edge[edge_index][k].end,
                                 safe_intervals.safe_intervals_edge[edge_index][k].end))
                    {
                        eq = false;
                        break;
                    }
                }

                if (!eq) break;
            }

            if (!eq) break;
        }
        if (eq)
        {
            node_1_ok = false;
            break;
        }
    }

    if (agent_id == 0) visited_1[visited_count_1++] = safe_intervals;
    else visited_2[visited_count_2++] = safe_intervals;
    
#endif
    size_t tmp_used = memory_arena->used;
    
    Path path_1 = sipp(memory_arena, graph,
                       agents[agent_id].start,
                       agents[agent_id].goal,
                       h, safe_intervals);
    if (path_1.vertex_count == 0)
    {
        node_1_ok = false;
    }
    else
    {
        Path tmp = path_buffer[agent_id];
        path_buffer[agent_id] = path_1;
        cost_1 = cost(path_buffer, agent_count);
        path_buffer[agent_id] = tmp;
    }

    memory_arena->used = tmp_used;

    if (node_1_ok)
    {
        result = push_struct(memory_arena, CBSNode);
        result->parent = node;
        result->constraint = constraint;
        result->cost = cost_1;
    }

    return result;
}

static Solution
cbs(MemoryArena* memory_arena, Graph* graph, AgentInfo* agents, u32 agent_count)
{
    Solution result;
    result.paths = push_array(memory_arena, Path, agent_count);
    result.path_count = agent_count;

    CBSNode* root = push_struct(memory_arena, CBSNode);
    
    CBSQueue* queue = 0;
    queue = add_cbs_node(memory_arena, queue, root);
    u32 queue_count = 1;

    visited_1 = push_array(memory_arena, ComputeSafeIntervalsResult, max_visited);
    visited_2 = push_array(memory_arena, ComputeSafeIntervalsResult, max_visited);
    
    while (queue_count > 0)
    {
#if 1
        {
            size_t tmp_used = memory_arena->used;
            CBSQueue* p = queue;
            CBSNode* _queue = push_array(memory_arena, CBSNode, queue_count);
            u32 i = 0;
            while (p) { _queue[i++] = *p->node; p = p->next; }
            memory_arena->used = tmp_used;
        }
#endif
        
        CBSNode* current_node = queue->node;
        queue = queue->next;
        queue_count--;
        
        size_t tmp_used = memory_arena->used;
        Path* path_buffer = push_array(memory_arena, Path, agent_count);

        b32 all_paths_valid = true;
        f32* path_costs = push_array(memory_arena, f32, agent_count);
        for (u32 agent_index = 0;
             agent_index < agent_count;
             agent_index++)
        {
            MakeConstraintResult constraints = make_constraints(memory_arena, current_node, agent_index);
            ComputeSafeIntervalsResult safe_intervals = compute_safe_intervals(memory_arena, graph, constraints.data, constraints.count);

            path_buffer[agent_index] = sipp(memory_arena, graph,
                                            agents[agent_index].start,
                                            agents[agent_index].goal,
                                            h, safe_intervals);
            path_costs[agent_index] = cost_path(path_buffer[agent_index]);
            if (path_buffer[agent_index].vertex_count == 0)
            {
                all_paths_valid = false;
                break;
            }
        }

        if (all_paths_valid)
        {
            FindConflictResult find_conflict_result = find_conflict(path_buffer, agent_count, graph);
            if (find_conflict_result.no_conflict_found)
            {
                Assert(!"done");
            }
            Conflict new_conflict = find_conflict_result.conflict;

            Constraint constraint_1 =  {new_conflict.interval,
                                        new_conflict.agent_1_id, new_conflict.action_1.type,
                                        new_conflict.action_1.move.from, new_conflict.action_1.move.to};
            CBSNode* node_1 = create_cbs_node(memory_arena, current_node,
                                              constraint_1, graph, agents,
                                              path_buffer, agent_count, new_conflict.agent_1_id);
            if (node_1)
            {
                queue = add_cbs_node(memory_arena, queue, node_1);
                queue_count++;
            }
            
            Constraint constraint_2 =  {new_conflict.interval,
                                        new_conflict.agent_2_id, new_conflict.action_2.type,
                                        new_conflict.action_2.move.from, new_conflict.action_2.move.to};
            CBSNode* node_2 = create_cbs_node(memory_arena, current_node,
                                              constraint_2, graph, agents,
                                              path_buffer, agent_count, new_conflict.agent_2_id);
            if (node_2)
            {
                queue = add_cbs_node(memory_arena, queue, node_2);
                queue_count++;
            }
        }
        else
        {
            memory_arena->used = tmp_used;
        }
    }
    
    return result;
}
    
struct GraphData
{
    Graph* graph;
    AgentInfo* agents;
    u32 agent_count;
};

static GraphData
load_graph(MemoryArena* memory_arena, const char* filename)
{
    GraphData graph_result;
    Graph* graph = push_struct(memory_arena, Graph);
    graph_result.graph = graph;

    DebugReadFileResult graph_file = DEBUG_platform_read_entire_file(filename);
    
    char* graph_raw = (char*)graph_file.contents;

    u32 graph_raw_index = 0;

    u32 agent_count = 0;
    u32 vertex_count = 0;
    skip_whitespace(&graph_raw_index, graph_raw);
    StringToU32Result agent_count_parse = string_to_u32(graph_raw, graph_raw_index);
    agent_count = agent_count_parse.number;
    graph_raw_index += agent_count_parse.length;
    while(graph_raw[graph_raw_index++] != ',');
    StringToU32Result vertex_count_parse = string_to_u32(graph_raw, graph_raw_index);
    vertex_count = vertex_count_parse.number;
    graph_raw_index += vertex_count_parse.length;
    
    u32 current_vertex = 0;
    u32 parsed_agents = 0;

    graph->vertex_count = vertex_count;
    graph->vertices = push_array(memory_arena, Vertex, vertex_count);
    graph_result.agents = push_array(memory_arena, AgentInfo, agent_count);
    graph_result.agent_count = agent_count;
        
    for (; graph_raw_index < graph_file.contents_size;
         graph_raw_index++)
    {
        if (IsCharacter(graph_raw[graph_raw_index]))
        {
            Assert(current_vertex < vertex_count);
            if (graph_raw[graph_raw_index] == 'a')
            {
                while(graph_raw[graph_raw_index++] != ',');
                skip_whitespace(&graph_raw_index, graph_raw);
                Assert(graph_raw[graph_raw_index] == 'n');
                graph_raw_index++;
                StringToU32Result vertex_parse = string_to_u32(graph_raw, graph_raw_index);
                graph_result.agents[parsed_agents].start = vertex_parse.number;
                graph_raw_index += vertex_parse.length;

                while(graph_raw[graph_raw_index++] != ',');
                skip_whitespace(&graph_raw_index, graph_raw);
                Assert(graph_raw[graph_raw_index] == 'n');
                graph_raw_index++;
                vertex_parse = string_to_u32(graph_raw, graph_raw_index);
                graph_result.agents[parsed_agents].goal = vertex_parse.number;
                graph_raw_index += vertex_parse.length;
                
                parsed_agents++;
            }
            else if (graph_raw[graph_raw_index] == 'n')
            {
                Vertex* vertex = &graph->vertices[current_vertex];
                
                while(graph_raw[graph_raw_index++] != ':');
                while(graph_raw[graph_raw_index++] != '(');
                skip_whitespace(&graph_raw_index, graph_raw);
                StringToU32Result x_parse = string_to_u32(graph_raw, graph_raw_index);
                vertex->x = (f32)x_parse.number;
                while(graph_raw[graph_raw_index++] != ',');
                skip_whitespace(&graph_raw_index, graph_raw);
                StringToU32Result y_parse = string_to_u32(graph_raw, graph_raw_index);
                vertex->y = (f32)y_parse.number;
                while(graph_raw[graph_raw_index++] != ')');
                skip_whitespace(&graph_raw_index, graph_raw);

                vertex->edges = push_struct(memory_arena, LinkedList);
                while(graph_raw[graph_raw_index] != ';')
                {
                    while(graph_raw[graph_raw_index++] != ',');
                    skip_whitespace(&graph_raw_index, graph_raw);
                    Assert(graph_raw[graph_raw_index] == 'n');
                    graph_raw_index++;
                    Edge* edge = push_struct(memory_arena, Edge);
                    StringToU32Result edge_parse = string_to_u32(graph_raw, graph_raw_index);
                    edge->to = edge_parse.number;
                    graph_raw_index += edge_parse.length;

                    while(graph_raw[graph_raw_index++] != ':');
                    StringToU32Result cost_parse = string_to_u32(graph_raw, graph_raw_index);
                    edge->cost = (f32)cost_parse.number;
                    graph_raw_index += cost_parse.length;
                    
                    vertex->edges->add(memory_arena, (u8*)edge, sizeof(Edge));
                    skip_whitespace(&graph_raw_index, graph_raw);
                }
                
                current_vertex++;
            }
        }
        else if (graph_raw[graph_raw_index] == ',' ||
                 IsWhitespace(graph_raw[graph_raw_index])){}
        else InvalidCodePath;
    }

    DEBUG_platform_free_file_memory(graph_file.contents);
    
    return graph_result;
}

struct SimulatorState
{
    b32 is_initialized;
    Graph* graph;
};

static void
simulate(MemoryArena* memory_arena)
{
    SimulatorState* simulator_state = (SimulatorState*)memory_arena->base;
    
    if (!simulator_state->is_initialized)
    {
        push_struct(memory_arena, SimulatorState);

        GraphData graph_data = load_graph(memory_arena, "graphs/grid4x4-2.grid");
        simulator_state->graph = graph_data.graph;
            
        simulator_state->is_initialized = true;

#if 0
        Conflicts* c1 = push_struct(memory_arena, Conflicts);
        c1->interval = {0, 1};
        c1->agent_1_id = 0;
        c1->action_1 = {{0, 1}, ACTION_TYPE_WAIT};
        c1->action_1.move = {1, 0};
        Conflicts* c2 = push_struct(memory_arena, Conflicts);
        c2->interval = {1, 2};
        c2->agent_1_id = 0;
        c2->action_1 = {{1, 2}, ACTION_TYPE_WAIT};
        c2->action_1.move = {1, 0};
        c1->next = c2;
        Conflicts* c3 = push_struct(memory_arena, Conflicts);
        c3->interval = {2, 3};
        c3->agent_1_id = 0;
        c3->action_1 = {{2, 3}, ACTION_TYPE_MOVE};
        c3->action_1.move = {1, 0};
        /* c2->next = c3; */
        Conflicts* c4 = push_struct(memory_arena, Conflicts);
        c4->interval = {3, 4};
        c4->agent_1_id = 0;
        c4->action_1 = {{3, 4}, ACTION_TYPE_MOVE};
        c4->action_1.move = {1, 0};
        c3->next = c4;
        Conflicts* c5 = push_struct(memory_arena, Conflicts);
        c5->interval = {4, 5};
        c5->agent_1_id = 0;
        c5->action_1 = {{4, 5}, ACTION_TYPE_MOVE};
        c5->action_1.move = {1, 0};
        c4->next = c5;
        Conflicts* c6 = push_struct(memory_arena, Conflicts);
        c6->interval = {5, 6};
        c6->agent_1_id = 0;
        c6->action_1 = {{5, 6}, ACTION_TYPE_MOVE};
        c6->action_1.move = {1, 0};
        c5->next = c6;
        Conflicts* c7 = push_struct(memory_arena, Conflicts);
        c7->interval = {6, 7};
        c7->agent_1_id = 0;
        c7->action_1 = {{6, 7}, ACTION_TYPE_MOVE};
        c7->action_1.move = {1, 0};
        c6->next = c7;
        
        sipp(memory_arena,
             graph_data.graph,
             1, 0,
             h,
             c1, 0);
        int x = 0;
#endif
        Solution solution = cbs(memory_arena, simulator_state->graph, graph_data.agents, graph_data.agent_count);
        /* Constraint* c2 = push_struct(memory_arena, Constraint); */
        /* c2->time = 2; */
        /* c2->vertex = 4; */
        /* constraints->add(memory_arena, (u8*)c2, sizeof(Constraint)); */

        /* Constraint* c4 = push_struct(memory_arena, Constraint); */
        /* c4->time = 3; */
        /* c4->vertex = 4; */
        /* constraints->add(memory_arena, (u8*)c4, sizeof(Constraint)); */
    }
}
