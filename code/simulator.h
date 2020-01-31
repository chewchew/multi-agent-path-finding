#if SIMULATOR_DEBUG
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#endif

#include <stdint.h>
#include <vector>
#include <queue>
#include <list>
#include <functional>

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

struct Edge
{
    u32 to;
    f32 cost;
};

struct Vertex
{
    f32 x;
    f32 y;
    std::vector<f32> speeds;
    std::vector<Edge> edges;
};

struct Graph
{
    std::vector<Vertex> vertices;
};

struct SIPPNode
{
    u32 vertex;
    f32 arrival_time;
    f32 safe_interval_end;
    f32 fscore;
    SIPPNode* parent;

    constexpr b32
    operator()(const SIPPNode*& node_1, const SIPPNode*& node_2) const 
        {
            return node_1->fscore < node_2->fscore;
        }

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
#define MAX_QUEUE_SIZE 100000

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
    b32 valid;
    CBSNode* parent;
    Constraint constraint;
    f32 cost;

    b32
    operator()(const CBSNode*& node_1, const CBSNode*& node_2) const 
        {
            return node_1->cost < node_2->cost;
        }
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

static void
add_astar_node(std::list<SIPPNode*>& queue, SIPPNode* node)
{
    auto it = queue.begin();
    while (it != queue.end())
    {
        if (node->fscore < (*it)->fscore)
        {
            break;
        }
        ++it;
    }
    queue.insert(it, node);
}

static void
create_sipp_node(SIPPNode* new_node,
                 SIPPNode* parent, u32 vertex,
                 f32 earliest_arrival_time, f32 safe_interval_end,
                 f32 move_time, f32 hvalue)
{
    new_node->parent = parent;
    new_node->vertex = vertex;
    new_node->arrival_time = earliest_arrival_time;
    new_node->safe_interval_end = safe_interval_end;
    new_node->fscore = earliest_arrival_time + hvalue;
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
can_wait_forever(std::vector<SafeInterval> safe_intervals, f32 from_time)
{
    b32 result = false;
    if (safe_intervals.size() == 0)
    {
        result = true;
    }
    else
    {
        for (SafeInterval interval : safe_intervals)
        {
            if (interval.start <= from_time && FloatEq(interval.end, INF))
            {
                result = true;
                break;
            }
        }
    }
    return result;
}

struct ComputeSafeIntervalsResult
{
    std::vector<std::vector<SafeInterval>> safe_intervals_vertex;
    std::vector<std::vector<SafeInterval>> safe_intervals_edge;
};

static void
remove_interval(std::vector<SafeInterval>& intervals, Interval conflict_interval)
{
    std::vector<SafeInterval> intervals_to_add;
    std::vector<u32> intervals_to_remove;
    std::vector<SafeInterval>::iterator it = intervals.begin();
    for (u32 safe_interval_index = 0;
           safe_interval_index < intervals.size();
           safe_interval_index++)
    {
        SafeInterval safe_interval = intervals[safe_interval_index];
        SafeInterval split_interval = intersection(conflict_interval, safe_interval);
        if (interval_exists(split_interval, false))
        {
            SafeInterval lower_interval = {safe_interval.start, split_interval.start};
            SafeInterval upper_interval = {split_interval.end, safe_interval.end};

            if (!interval_exists(lower_interval, false))
            {
                if (!interval_exists(upper_interval, false))
                {
                    intervals_to_remove.push_back(safe_interval_index);
                }
                else
                {
                    intervals_to_remove.push_back(safe_interval_index);
                    intervals_to_add.push_back(upper_interval);
                }
            }
            else if (!interval_exists(upper_interval, false))
            {
                intervals_to_remove.push_back(safe_interval_index);
                intervals_to_add.push_back(lower_interval);
            }
            else
            {
                intervals_to_remove.push_back(safe_interval_index);
                intervals_to_add.push_back(lower_interval);
                intervals_to_add.push_back(upper_interval);
            }
        }
    };

    for (u32 remove_index : intervals_to_remove)
    {
        intervals.erase(intervals.begin() + remove_index);
    }
    
    intervals.insert(intervals.begin(), intervals_to_add.begin(), intervals_to_add.end());
}

static u32
get_edge_index(u32 vertex_count, u32 from, u32 to)
{
    return vertex_count * from + to;
}

#define GetNode(pool_id, count) &pool_id[count++]

static std::vector<SIPPNode>
sipp(Graph* graph,
     u32 start, u32 goal,
     f32 (*heuristic)(Graph*, u32, u32),
     ComputeSafeIntervalsResult safe_intervals)
{
    std::vector<std::vector<SafeInterval>> safe_intervals_vertex = safe_intervals.safe_intervals_vertex;
    std::vector<std::vector<SafeInterval>> safe_intervals_edge = safe_intervals.safe_intervals_edge;
    SIPPNode* node_pool = (SIPPNode*)malloc(sizeof(SIPPNode) * MAX_QUEUE_SIZE);
    u32 nodes_in_use = 0;
    
    std::vector<SIPPNode> result;

    std::list<SIPPNode*> queue;
    SIPPNode* root = GetNode(node_pool, nodes_in_use);
    f32 root_safe_interval_end = safe_intervals_vertex[start][0].end;
    if (safe_intervals_vertex[start][0].start > 0)
    {
        root_safe_interval_end = 0;
    }
    create_sipp_node(root,
                     0, start,
                     0, root_safe_interval_end,
                     0, heuristic(graph, start, goal));
    add_astar_node(queue, root);
    

    std::vector<std::vector<f32>> visited;
    visited.resize(graph->vertices.size());
    
    u32 expansions = 0;
    while(queue.size() > 0)
    {        
        expansions++;
        SIPPNode* current_node = queue.front();
        queue.pop_front();
        
        u32 current_vertex = current_node->vertex;
        f32 arrival_time = current_node->arrival_time;
        f32 safe_interval_end = current_node->safe_interval_end;
        
        Assert(current_vertex < graph->vertices.size());
        
        if (current_vertex == goal && can_wait_forever(safe_intervals_vertex[current_vertex], arrival_time))
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
            
            next = current_node;
            for (u32 vertex_index = 0;
                 vertex_index < path_length;
                 vertex_index++)
            {
                result.push_back(path[path_length - 1 - vertex_index]);
            }
            
            break;
        }

        std::vector<Edge> neighbours = graph->vertices[current_vertex].edges;
        for (Edge edge : neighbours)
        {
            f32 neighbour_cost = edge.cost;
            u32 neighbour_to = edge.to;
            f32 hvalue = heuristic(graph, neighbour_to, goal);

            f32 earliest_departure_time = arrival_time;
            f32 earliest_arrival_time = earliest_departure_time + neighbour_cost;
            f32 latest_departure_time = safe_interval_end;
            f32 latest_arrival_time = latest_departure_time + neighbour_cost;
            SafeInterval arrival_interval = {earliest_arrival_time, latest_arrival_time};

            for (SafeInterval vertex_interval : safe_intervals_vertex[neighbour_to])
            {
                SafeInterval safe_arrival_interval_vertex = intersection(arrival_interval, vertex_interval);
                if (interval_exists(safe_arrival_interval_vertex))
                {
                    SafeInterval safe_departure_interval_vertex = {safe_arrival_interval_vertex.start - neighbour_cost,
                                                                   safe_arrival_interval_vertex.end - neighbour_cost};
                    u32 edge_index = get_edge_index(graph->vertices.size(), current_vertex,  neighbour_to);
                    for (SafeInterval edge_interval : safe_intervals_edge[edge_index])
                    {                            
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

                        auto el = std::find(visited[neighbour_to].begin(), visited[neighbour_to].end(), earliest_arrival_time);
                        if (el == visited[neighbour_to].end())
                        {
                            SIPPNode* new_node = GetNode(node_pool, nodes_in_use);
                            create_sipp_node(new_node,
                                             current_node, neighbour_to,
                                             earliest_arrival_time, vertex_interval.end,
                                             neighbour_cost, hvalue);
                            add_astar_node(queue, new_node);

                            visited[neighbour_to].push_back(earliest_arrival_time);
                        }
                    }
                }
            }
        }
    }

    free(node_pool);
    
    return result;
}

struct AgentInfo
{
    u32 id;
    u32 start;
    u32 goal;
};

struct Solution
{
    std::vector<std::vector<SIPPNode>> paths;
    u32 path_count;
};

static ComputeSafeIntervalsResult
compute_safe_intervals(Graph* graph, std::vector<Constraint> constraints)
{
    ComputeSafeIntervalsResult result;
    
    u32 safe_intervals_vertex_size = graph->vertices.size();
    result.safe_intervals_vertex.resize(safe_intervals_vertex_size);
    for (u32 safe_interval_index = 0;
         safe_interval_index < safe_intervals_vertex_size;
         safe_interval_index++)
    {
        result.safe_intervals_vertex[safe_interval_index].push_back({0, INF});
    }
    u32 safe_intervals_edge_size = graph->vertices.size() * graph->vertices.size();
    result.safe_intervals_edge.resize(safe_intervals_edge_size);
    for (u32 safe_interval_index = 0;
         safe_interval_index < safe_intervals_edge_size;
         safe_interval_index++)
    {
        result.safe_intervals_edge[safe_interval_index].push_back({0, INF});
    }
    
    for (Constraint constraint : constraints)
    {
        // TODO: ta bort kollisions-intervallet från safe intervals
        if (constraint.type == ACTION_TYPE_WAIT)
        {
            u32 vertex_index = constraint.from;
            Interval interval = constraint.interval;
                
            remove_interval(result.safe_intervals_vertex[vertex_index], interval);
        }
        else if (constraint.type == ACTION_TYPE_MOVE)
        {
            u32 from_vertex_index = constraint.from;
            u32 to_vertex_index = constraint.to;
            Interval interval = constraint.interval;

            u32 edge_index = get_edge_index(graph->vertices.size(), from_vertex_index, to_vertex_index);
                
            remove_interval(result.safe_intervals_edge[edge_index], interval);
        }
    };

    return result;
}

f32 h(Graph* g, u32 vertex, u32 goal)
{
    f32 x = g->vertices[vertex].x - g->vertices[goal].x;
    f32 y = g->vertices[vertex].y - g->vertices[goal].y;
    
    return sqrtf(x * x + y * y);
}        

static f32
cost_path(std::vector<SIPPNode> path)
{
    return path[path.size() - 1].arrival_time;
}

static f32
cost(std::vector<std::vector<SIPPNode>> paths)
{
    f32 result = 0;

    for (std::vector<SIPPNode> path : paths)
    {
        result += cost_path(path);
    }
    
    return result;
}

static void
add_cbs_node(std::list<CBSNode*>& queue, CBSNode* node)
{
    auto it = queue.begin();
    while (it != queue.end())
    {
        if (node->cost < (*it)->cost)
        {
            break;
        }
        ++it;
    }
    queue.insert(it, node);
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
        std::vector<Edge> neighbours = graph->vertices[vertex.vertex].edges;
        f32 edge_cost = -1;
        for (Edge neighbour : neighbours)
        {
            if (neighbour.to == next_vertex.vertex)
            {
                edge_cost = neighbour.cost;
                break;
            }
        }
        Assert(edge_cost > 0);
        departure_time_vertex = next_vertex.arrival_time - edge_cost;
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
find_conflict(std::vector<std::vector<SIPPNode>> path_buffer, Graph* graph)
{
    FindConflictResult result = {};
    result.no_conflict_found = true;

    u32 agent_count = path_buffer.size();
    for (u32 agent_index = 0;
         agent_index < agent_count && result.no_conflict_found;
         agent_index++)
    {
        std::vector<SIPPNode> path = path_buffer[agent_index];
        for (u32 vertex_index = 0;
             vertex_index < path.size() && result.no_conflict_found;
             vertex_index++)
        {
            SIPPNode vertex = path[vertex_index];
            SIPPNode next_vertex = path[(vertex_index + 1) % path.size()];
            b32 last_vertex = vertex_index == path.size() - 1;
            ActionIntervals action_intervals_vertex = create_action_intervals(vertex, last_vertex, next_vertex, graph);
            
            for (u32 other_agent_index = agent_index + 1;
                 other_agent_index < agent_count && result.no_conflict_found;
                 other_agent_index++)
            {
                std::vector<SIPPNode> other_path = path_buffer[other_agent_index];
                for (u32 other_vertex_index = 0;
                     other_vertex_index < other_path.size() && result.no_conflict_found;
                     other_vertex_index++)
                {
                    SIPPNode other_vertex = other_path[other_vertex_index];
                    SIPPNode next_other_vertex = other_path[(other_vertex_index + 1) % other_path.size()];
                    b32 last_other_vertex = other_vertex_index == other_path.size() - 1;
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
    
static std::vector<Constraint>
make_constraints(CBSNode* node, u32 agent_id)
{
    std::vector<Constraint> result;

    CBSNode* p = node;
    while (p->parent)
    {
        if (p->constraint.agent_id == agent_id)
        {
            result.push_back(p->constraint);
        }
        p = p->parent;
    }

    return result;
}

#if 0
global_variable u32 max_visited = 100000;
global_variable u32* visited_count;
global_variable MakeConstraintResult** visited;
#endif

static b32
constraint_sets_equal(std::vector<Constraint> set_1, std::vector<Constraint> set_2)
{
    b32 result = false;
    
    if (set_1.size() == set_2.size())
    {
        std::vector<u32> checked;
        checked.resize(set_1.size());
        for (u32 constraint_index = 0;
             constraint_index < set_1.size();
             constraint_index++)
        {
            if (!checked[constraint_index])
            {
                Constraint c1 = set_1[constraint_index];
                Constraint c2 = set_2[constraint_index];
                if (constraints_equal(c1, c2))
                {
                    checked[constraint_index] = true;
                }
            }
        }

        result = true;
        for (u32 constraint_index = 0;
             constraint_index < set_1.size();
             constraint_index++)
        {
            if (!checked[constraint_index])
            {
                result = false;
                break;
            }
        }
    }

    return result;
}

static void
create_cbs_node(CBSNode* new_node,
                CBSNode* parent, std::vector<std::vector<std::vector<Constraint>>> visited,
                Constraint constraint, Graph* graph, std::vector<AgentInfo> agents,
                std::vector<std::vector<SIPPNode>> path_buffer, u32 agent_id)
{
    new_node->parent = parent;
    new_node->constraint = constraint;
    std::vector<Constraint> constraints = make_constraints(new_node, agent_id);
    for (std::vector<Constraint> constraint_set : visited[agent_id])
    {
        if (constraint_sets_equal(constraint_set, constraints))
        {
            new_node->valid = false;
            return;
        }
    }
    visited[agent_id].push_back(constraints);
    
    ComputeSafeIntervalsResult safe_intervals = compute_safe_intervals(graph, constraints);

    std::vector<SIPPNode> new_path = sipp(graph,
                                        agents[agent_id].start,
                                        agents[agent_id].goal,
                                        h, safe_intervals);
    if (new_path.size() == 0)
    {
        new_node->valid = false;
        return;
    }

    new_node->parent = parent;
    new_node->constraint = constraint;
    new_node->cost = cost(path_buffer) - cost_path(path_buffer[agent_id]) + cost_path(new_path);
}

static Solution
cbs(Graph* graph, std::vector<AgentInfo> agents)
{
    Solution result;
    result.paths.resize(agents.size());

    CBSNode* node_pool = (CBSNode*)malloc(sizeof(CBSNode) * MAX_QUEUE_SIZE);
    u32 nodes_in_use = 0;

    std::list<CBSNode*> queue;
    std::vector<std::vector<std::vector<Constraint>>> visited;
    visited.resize(agents.size());
    std::vector<std::vector<SIPPNode>> path_buffer;
    path_buffer.resize(agents.size());

    CBSNode* root = GetNode(node_pool, nodes_in_use);
    root->valid = true;
    root->parent = 0;
    root->constraint = {};
    root->cost = 0;
    
    add_cbs_node(queue, root);
    
    while (queue.size() > 0)
    {
        CBSNode* current_node = queue.front();
        queue.pop_front();

        b32 all_paths_valid = true;
        for (AgentInfo agent : agents)
        {
            std::vector<Constraint> constraints = make_constraints(current_node, agent.id);
            ComputeSafeIntervalsResult safe_intervals = compute_safe_intervals(graph, constraints);

            path_buffer[agent.id] = sipp(graph,
                                         agent.start,
                                         agent.goal,
                                         h, safe_intervals);
            if (path_buffer[agent.id].size() == 0)
            {
                all_paths_valid = false;
                break;
            }
        }

        if (all_paths_valid)
        {
            FindConflictResult find_conflict_result = find_conflict(path_buffer, graph);
            if (find_conflict_result.no_conflict_found)
            {
                Assert(!"done");
            }
            Conflict new_conflict = find_conflict_result.conflict;

            Constraint constraint_1 =  {new_conflict.interval,
                                        new_conflict.agent_1_id, new_conflict.action_1.type,
                                        new_conflict.action_1.move.from, new_conflict.action_1.move.to};
            CBSNode* node_1 = GetNode(node_pool, nodes_in_use);
            create_cbs_node(node_1,
                            current_node, visited,
                            constraint_1, graph, agents,
                            path_buffer, new_conflict.agent_1_id);
            if (node_1)
            {
                add_cbs_node(queue, node_1);
            }
            else
            {
                nodes_in_use--;
            }
            
            Constraint constraint_2 =  {new_conflict.interval,
                                        new_conflict.agent_2_id, new_conflict.action_2.type,
                                        new_conflict.action_2.move.from, new_conflict.action_2.move.to};
            CBSNode* node_2 = GetNode(node_pool, nodes_in_use);
            create_cbs_node(node_2,
                            current_node, visited,
                            constraint_2, graph, agents,
                            path_buffer, new_conflict.agent_2_id);
            if (node_2)
            {
                add_cbs_node(queue, node_2);
            }
            else
            {
                nodes_in_use--;
            }
        }
    }

    free(node_pool);
    
    return result;
}
    
struct GraphData
{
    Graph graph;
    std::vector<AgentInfo> agents;
};

static GraphData
load_graph(const char* filename)
{
    GraphData graph_result;

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
    graph_result.agents.resize(agent_count);
    
    u32 current_vertex = 0;
    u32 parsed_agents = 0;

    graph_result.graph.vertices.resize(vertex_count);
        
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
                
                graph_result.agents[parsed_agents].id = parsed_agents;
                parsed_agents++;
            }
            else if (graph_raw[graph_raw_index] == 'n')
            {
                Vertex* vertex = &graph_result.graph.vertices[current_vertex];
                
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

                /* vertex->edges ; */
                while(graph_raw[graph_raw_index] != ';')
                {
                    while(graph_raw[graph_raw_index++] != ',');
                    skip_whitespace(&graph_raw_index, graph_raw);
                    Assert(graph_raw[graph_raw_index] == 'n');
                    graph_raw_index++;
                    Edge edge;
                    StringToU32Result edge_parse = string_to_u32(graph_raw, graph_raw_index);
                    edge.to = edge_parse.number;
                    graph_raw_index += edge_parse.length;

                    while(graph_raw[graph_raw_index++] != ':');
                    StringToU32Result cost_parse = string_to_u32(graph_raw, graph_raw_index);
                    edge.cost = (f32)cost_parse.number;
                    graph_raw_index += cost_parse.length;
                    
                    vertex->edges.push_back(edge);
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
simulate(SimulatorState* simulator_state)
{
    if (!simulator_state->is_initialized)
    {
        GraphData graph_data = load_graph("graphs/grid3x3.grid");
            
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
        Solution solution = cbs(&graph_data.graph, graph_data.agents);
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
