
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#include "vec2.h"

struct Edge
{
    u32 to;
    f32 cost;
};
    
struct Vertex
{
    u32 id;
    Vec2 position;
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
    f32 speed;
    SIPPNode* parent;
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

#define UNDEFINED_INTERVAL {INF, INF}

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

static SIPPNode
create_ds_sipp_node(SIPPNode* parent, u32 vertex,
                    f32 earliest_arrival_time, f32 safe_interval_end,
                    f32 move_time, f32 hvalue, f32 speed)
{
    SIPPNode new_node;
    new_node.parent = parent;
    new_node.vertex = vertex;
    new_node.arrival_time = earliest_arrival_time;
    new_node.safe_interval_end = safe_interval_end;
    new_node.fscore = earliest_arrival_time + hvalue;
    new_node.speed = speed;
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

static std::vector<SIPPNode*>
get_successors_sipp(Graph* graph,
                    SIPPNode* current_node, u32 goal,
                    f32 (*heuristic)(Graph*, u32, u32),
                    ComputeSafeIntervalsResult safe_intervals,
                    std::vector<std::vector<f32>> visited,
                    std::vector<SIPPNode>& node_pool)
{
    std::vector<SIPPNode*> successors;
    
    u32 current_vertex = current_node->vertex;
    f32 arrival_time = current_node->arrival_time;
    f32 safe_interval_end = current_node->safe_interval_end;
    std::vector<std::vector<SafeInterval>> safe_intervals_vertex = safe_intervals.safe_intervals_vertex;
    std::vector<std::vector<SafeInterval>> safe_intervals_edge = safe_intervals.safe_intervals_edge;
        
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
                        node_pool.push_back({});
                        SIPPNode* new_node = &node_pool[node_pool.size() - 1];
                        create_sipp_node(new_node,
                                         current_node, neighbour_to,
                                         earliest_arrival_time, vertex_interval.end,
                                         neighbour_cost, hvalue);
                        successors.push_back(new_node);
                    }
                }
            }
        }
    }

    return successors;
}

static b32
action_possible(Vertex* prev, Vertex current, f32 current_speed, Vertex next, f32 next_speed)
{
    if (FloatEq(current_speed, 1) && FloatEq(next_speed, 0))
    {
        return false;
    }
    else
    {
        return true;
    }
}

static f32
action_time(Vertex current, f32 current_speed, Vertex next, f32 next_speed)
{
    f32 d = distance(current.position, next.position);
    return 2 * d / (current_speed + next_speed);
}

static std::vector<SIPPNode>
get_successors_ds_sipp(Graph* graph,
                       SIPPNode* current_node, u32 goal,
                       f32 (*heuristic)(Graph*, u32, u32),
                       ComputeSafeIntervalsResult safe_intervals,
                       std::vector<std::vector<f32>> visited)
{
    std::vector<SIPPNode> successors;

    u32 current_vertex_id = current_node->vertex;
    Vertex current_vertex = graph->vertices[current_vertex_id];
    f32 current_speed = current_node->speed;
    Vertex* prev = 0;
    if (current_node->parent)
    {
        prev = &graph->vertices[current_node->parent->vertex];
    }
    
    std::vector<std::vector<SafeInterval>> safe_intervals_vertex = safe_intervals.safe_intervals_vertex;
    std::vector<std::vector<SafeInterval>> safe_intervals_edge = safe_intervals.safe_intervals_edge;
        
    std::vector<Edge> neighbours = current_vertex.edges;
    for (Edge edge : neighbours)
    {
        u32 neighbour_to = edge.to;
        Vertex neighbour = graph->vertices[neighbour_to];
        f32 hvalue = heuristic(graph, neighbour_to, goal);

        for (f32 neighbour_speed : neighbour.speeds)
        {
            if (action_possible(prev, current_vertex, current_speed, neighbour, neighbour_speed))
            {
                f32 neighbour_cost = action_time(current_vertex, current_speed, neighbour, neighbour_speed);

                f32 earliest_departure_time;
                f32 latest_departure_time;
                if (FloatEq(current_speed, 0))
                {
                    earliest_departure_time = current_node->arrival_time;
                    latest_departure_time = current_node->safe_interval_end;
                }
                else
                {
                    earliest_departure_time = latest_departure_time = current_node->arrival_time;
                }
                f32 earliest_arrival_time = earliest_departure_time + neighbour_cost;
                f32 latest_arrival_time = latest_departure_time + neighbour_cost;
                SafeInterval arrival_interval = {earliest_arrival_time, latest_arrival_time};

                for (SafeInterval vertex_interval : safe_intervals_vertex[neighbour_to])
                {
                    SafeInterval safe_arrival_interval_vertex = intersection(arrival_interval, vertex_interval);
                    if (interval_exists(safe_arrival_interval_vertex))
                    {
                        SafeInterval safe_departure_interval_vertex = {safe_arrival_interval_vertex.start - neighbour_cost,
                                                                       safe_arrival_interval_vertex.end - neighbour_cost};
                        u32 edge_index = get_edge_index(graph->vertices.size(), current_node->vertex, neighbour_to);
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
                                successors.push_back(
                                    create_ds_sipp_node(current_node, neighbour_to,
                                                        earliest_arrival_time, vertex_interval.end,
                                                        neighbour_cost, hvalue, neighbour_speed)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    return successors;
}

static std::vector<SIPPNode>
sipp(Graph* graph,
     u32 start_vertex, f32 start_speed, u32 goal_vertex, f32 goal_speed,
     f32 (*heuristic)(Graph*, u32, u32),
     ComputeSafeIntervalsResult safe_intervals)
{
    std::vector<SIPPNode> node_pool;
    node_pool.resize(MAX_QUEUE_SIZE);
    u32 node_count = 0;
    
    std::vector<SIPPNode> result;

    std::list<SIPPNode*> queue;
    
    f32 root_safe_interval_end = safe_intervals.safe_intervals_vertex[start_vertex][0].end;
    if (safe_intervals.safe_intervals_vertex[start_vertex][0].start > 0)
    {
        root_safe_interval_end = 0;
    }
    SIPPNode root = create_ds_sipp_node(0, start_vertex,
                                        0, root_safe_interval_end,
                                        0, heuristic(graph, start_vertex, goal_vertex), start_speed);
    SIPPNode* root_node = &node_pool[node_count++];
    *root_node = root;
    add_astar_node(queue, root_node);
    

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
        
        Assert(current_vertex < graph->vertices.size());
        
        if (current_vertex == goal_vertex && FloatEq(current_node->speed, goal_speed) && can_wait_forever(safe_intervals.safe_intervals_vertex[current_vertex], arrival_time))
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

        std::vector<SIPPNode> successors = get_successors_ds_sipp(graph, current_node, goal_vertex,
                                                                   heuristic, safe_intervals,
                                                                   visited);
        for (SIPPNode successor : successors)
        {
            SIPPNode* new_node = &node_pool[node_count++];
            *new_node = successor;
            add_astar_node(queue, new_node);
            visited[successor.vertex].push_back(successor.arrival_time);
            node_count++;
        }
    }
    
    return result;
}

struct AgentInfo
{
    u32 id;

    u32 start_vertex;
    f32 start_speed;
    u32 goal_vertex;
    f32 goal_speed;
    
    f32 radius;
};

struct Solution
{
    std::vector<std::vector<SIPPNode>> paths;
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
    return distance(g->vertices[vertex].position, g->vertices[goal].position);
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

// Source: Guide to anticipatory collision avoidance Chapter 19
struct Collision
{
    f32 t;
    b32 ok;
};

static Collision
get_collision_time(Vec2 pos_1, f32 radius_1, Vec2 velocity_1,
                   Vec2 pos_2, f32 radius_2, Vec2 velocity_2)
{
    Collision result = {};
    
    f32 t_pos = 0;
    f32 t_neg = 0;
    
    Vec2 w = pos_2 - pos_1;
    Vec2 v = velocity_2 - velocity_1;
    f32 a = dot(v, v);
    f32 b = dot(w, velocity_1 - velocity_2);
    f32 radius_sum = radius_1 + radius_2;
    f32 c = dot(w, w) - radius_sum * radius_sum;
    
    f32 term_1 = b * b;
    f32 term_2 = a * c;
    if (a > 0 && term_1 >= term_2)
    {
        t_pos = (b + sqrtf(term_1 - term_2)) / a;
        t_neg = (b - sqrtf(term_1 - term_2)) / a;

        result.ok = true;
    }
    else
    {
        result.ok = false;
    }

    result.t = fmin(t_pos, t_neg);
    
    return result;
}

static Interval
get_collision_interval(Vec2 pos_1, f32 radius_1, Vec2 velocity_1,
                       Vec2 pos_2, f32 radius_2, Vec2 velocity_2,
                       Interval action_interval)
{
    Interval result = UNDEFINED_INTERVAL;
    
    Collision c = get_collision_time(pos_1, radius_1, velocity_1,
                               pos_2, radius_2, velocity_2);
    f32 t = c.t;
    
    if (c.ok && (t > 0 || FloatEq(t, 0)) && action_interval.start + t < action_interval.end)
    {
        Vec2 search_pos_1 = pos_1 + velocity_1 * t;
        Vec2 search_pos_2 = pos_2 + velocity_2 * t;
            
        result.start = action_interval.start + t;

        // NOTE: Collision detect until no collision = result.end
        f32 delta = 0.01f;
        while (action_interval.start + t < action_interval.end)
        {
            t += delta;
            search_pos_1 += velocity_1 * delta;
            search_pos_2 += velocity_2 * delta;

            f32 dist = distance(search_pos_1, search_pos_2);
            if (dist > radius_1 + radius_2)
            {
                break;
            }
        }

        result.end = result.start + t;
    }

    return result;
}

struct FindConflictResult
{
    b32 no_conflict_found;
    Conflict conflict;
};

static FindConflictResult
find_conflict(std::vector<std::vector<SIPPNode>> path_buffer, Graph* graph, std::vector<AgentInfo> agents)
{
    FindConflictResult result = {};
    result.no_conflict_found = true;

    u32 agent_count = path_buffer.size();
    for (u32 agent_index = 0;
         agent_index < agent_count && result.no_conflict_found;
         agent_index++)
    {
        std::vector<SIPPNode> path = path_buffer[agent_index];
        for (u32 node_index = 0;
             node_index < path.size() - 1 && result.no_conflict_found;
             node_index++)
        {
            SIPPNode node = path[node_index];
            Vertex vertex = graph->vertices[node.vertex];
            SIPPNode next_node = path[(node_index + 1) % path.size()];
            Vertex next_vertex = graph->vertices[next_node.vertex];
            b32 last_node = false;//node_index == path.size() - 1;
            ActionIntervals action_intervals_node = create_action_intervals(node, last_node, next_node, graph);
            AgentInfo agent_1 = agents[agent_index];
            
            for (u32 other_agent_index = agent_index + 1;
                 other_agent_index < agent_count && result.no_conflict_found;
                 other_agent_index++)
            {
                std::vector<SIPPNode> other_path = path_buffer[other_agent_index];
                for (u32 other_node_index = 0;
                     other_node_index < other_path.size() - 1 && result.no_conflict_found;
                     other_node_index++)
                {
                    SIPPNode other_node = other_path[other_node_index];
                    Vertex other_vertex = graph->vertices[other_node.vertex];
                    SIPPNode next_other_node = other_path[(other_node_index + 1) % other_path.size()];
                    Vertex next_other_vertex = graph->vertices[next_other_node.vertex];
                    b32 last_other_node = false; //other_node_index == other_path.size() - 1;
                    ActionIntervals action_intervals_other_node = create_action_intervals(other_node, last_other_node, next_other_node, graph);
                    AgentInfo agent_2 = agents[other_agent_index];

                    Interval move_move_interval = intersection(action_intervals_node.move, action_intervals_other_node.move);                    
                    b32 move_move_interval_ok =
                        move_move_interval.start != move_move_interval.end &&
                        interval_exists(move_move_interval);

                    Interval move_wait_interval = intersection(action_intervals_node.move, action_intervals_other_node.wait);
                    b32 move_wait_interval_ok =
                        move_wait_interval.start != move_wait_interval.end &&
                        interval_exists(move_wait_interval);

                    Interval wait_move_interval = intersection(action_intervals_node.wait, action_intervals_other_node.move);
                    b32 wait_move_interval_ok =
                        wait_move_interval.start != wait_move_interval.end &&
                        interval_exists(wait_move_interval);
                    
                    Vec2 agent_1_pos = vertex.position;
                    Vec2 agent_2_pos = other_vertex.position;
                        
                    Vec2 velocity_1 = next_vertex.position - vertex.position;
                    Vec2 velocity_2 = next_other_vertex.position - other_vertex.position;
                    Vec2 zero_velocity = {};
                    
                    if (move_move_interval_ok)
                    {    
                        Interval collision_interval = get_collision_interval(
                            agent_1_pos, agent_1.radius, velocity_1,
                            agent_2_pos, agent_2.radius, velocity_2,
                            move_move_interval
                        );

                        if (interval_exists(collision_interval))
                        {    
                            result.no_conflict_found = false;
                            result.conflict.interval = collision_interval;

                            result.conflict.agent_1_id = agent_index;
                            result.conflict.action_1 = {action_intervals_node.move, ACTION_TYPE_MOVE};
                            result.conflict.action_1.move.from = node.vertex;
                            result.conflict.action_1.move.to = next_node.vertex;

                            result.conflict.agent_2_id = other_agent_index;
                            result.conflict.action_2 = {action_intervals_other_node.move, ACTION_TYPE_MOVE};
                            result.conflict.action_2.move.from = other_node.vertex;
                            result.conflict.action_2.move.to = next_other_node.vertex;
                        }
                    }
                    else if (move_wait_interval_ok)
                    {
                        Interval collision_interval = get_collision_interval(
                            agent_1_pos, agent_1.radius, velocity_1,
                            agent_2_pos, agent_2.radius, zero_velocity,
                            move_wait_interval
                        );

                        if (interval_exists(collision_interval))
                        {
                            result.no_conflict_found = false;
                            result.conflict.interval = collision_interval;

                            result.conflict.agent_1_id = agent_index;
                            result.conflict.action_1 = {action_intervals_node.move, ACTION_TYPE_MOVE};
                            result.conflict.action_1.move.from = node.vertex;
                            result.conflict.action_1.move.to = next_node.vertex;

                            result.conflict.agent_2_id = other_agent_index;
                            result.conflict.action_2 = {action_intervals_other_node.wait, ACTION_TYPE_WAIT, other_node.vertex};
                        }
                    }
                    else if (wait_move_interval_ok)
                    {
                        Interval collision_interval = get_collision_interval(
                            agent_1_pos, agent_1.radius, zero_velocity,
                            agent_2_pos, agent_2.radius, velocity_2,
                            wait_move_interval
                        );

                        if (interval_exists(collision_interval))
                        {
                            result.no_conflict_found = false;
                            result.conflict.interval = collision_interval;

                            result.conflict.agent_1_id = agent_index;
                            result.conflict.action_1 = {action_intervals_node.wait, ACTION_TYPE_WAIT, node.vertex};

                            result.conflict.agent_2_id = other_agent_index;
                            result.conflict.action_2 = {action_intervals_other_node.move, ACTION_TYPE_MOVE};
                            result.conflict.action_2.move.from = other_node.vertex;
                            result.conflict.action_2.move.to = next_other_node.vertex;
                        }
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
                                          agents[agent_id].start_vertex, agents[agent_id].start_speed,
                                          agents[agent_id].goal_vertex, agents[agent_id].goal_speed,
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
                                         agent.start_vertex, agent.start_speed,
                                         agent.goal_vertex, agent.goal_speed,
                                         h, safe_intervals);
            if (path_buffer[agent.id].size() == 0)
            {
                all_paths_valid = false;
                break;
            }
        }

        if (all_paths_valid)
        {
            FindConflictResult find_conflict_result = find_conflict(path_buffer, graph, agents);
            if (find_conflict_result.no_conflict_found)
            {
                result.paths = path_buffer;
                break;
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

static void
simulate(Graph graph, std::vector<AgentInfo> agents)
{    
    Solution solution = cbs(&graph, agents);
    int x = 0;
}
