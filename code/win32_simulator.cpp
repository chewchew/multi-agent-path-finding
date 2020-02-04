#include "simulator.h"

struct GraphData
{
    Graph graph;
    std::vector<AgentInfo> agents;
};

#define LANE_WIDTH 0.5f
#define MIN_COLUMN_WIDTH 0.5f
#define AGENT_RADIUS 0.4f
// agent_count = number of agents
// road_length = number of waypoints in road
static GraphData
get_random_road_ds(u32 agent_count, u32 road_length)
{
    GraphData result;
    result.graph.vertices.resize(2 * road_length);

    std::vector<f32> speeds = {0, 0.25, 0.5, 0.75, 1};

    u32 left_goal_index = road_length - 1;
    u32 right_goal_index = 2 * road_length - 1;
    result.graph.vertices[left_goal_index] = {left_goal_index, {left_goal_index * MIN_COLUMN_WIDTH, 0}, speeds};
    result.graph.vertices[right_goal_index] = {right_goal_index, {left_goal_index * MIN_COLUMN_WIDTH, 1}, speeds};
    for (u32 column_index = 0;
         column_index < road_length - 1;
         column_index++)
    {
        u32 slot_1 = column_index;
        u32 slot_2 = road_length + column_index;
        
        result.graph.vertices[slot_1] = {slot_1, {column_index * MIN_COLUMN_WIDTH, 0}, speeds};
        result.graph.vertices[slot_2] = {slot_2, {column_index * MIN_COLUMN_WIDTH, 1}, speeds};

        u32 forward_slot_1 = slot_1 + 1;
        u32 forward_slot_2 = slot_2 + 1;
        result.graph.vertices[slot_1].edges.push_back({forward_slot_1, 1});
        result.graph.vertices[slot_2].edges.push_back({forward_slot_2, 1});

        u32 switch_slot_1 = road_length + column_index + 1;
        u32 switch_slot_2 = column_index + 1;
        result.graph.vertices[slot_1].edges.push_back({switch_slot_1, 1});
        result.graph.vertices[slot_2].edges.push_back({switch_slot_2, 1});
    }

    b32* occupied = (b32*)malloc(sizeof(b32) * 2 * road_length);
    for (u32 i = 0; i < 2 * road_length; i++) occupied[i] = false;
    
    for (u32 agent_index = 0;
         agent_index < agent_count;
         agent_index++)
    {
        u32 max_vertex = road_length - 2;
        u32 start_vertex = rand() % max_vertex;
        while (occupied[start_vertex]) start_vertex = rand() % max_vertex;
        occupied[start_vertex] = true;
        u32 goal_vertex;
        if (rand() % 2 == 0)
        {
            goal_vertex = left_goal_index;
        }
        else
        {
            goal_vertex = right_goal_index;
        }

        f32 start_speed = speeds[rand() % speeds.size()];
        f32 goal_speed = speeds[rand() % speeds.size()];
        result.agents.push_back({agent_index, start_vertex, start_speed, goal_vertex, goal_speed, AGENT_RADIUS});
    }

    free(occupied);

    return result;
}

int main()
{
    GraphData data = get_random_road_ds(5, 7);
    simulate(data.graph, data.agents);
}
