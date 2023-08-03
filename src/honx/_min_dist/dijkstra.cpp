//
// Created by honglu on 8/3/23.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <vector>
#include <queue>

namespace py = pybind11;

std::vector<int> dijkstra_stl(const std::vector<std::vector<std::pair<int, int>>>& adj_list, int start){
    // initialize the distance vector
    std::vector<int> dist(adj_list.size(), INT_MAX);
    dist[start] = 0;

    // initialize the priority queue
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, start});

    // the main loop
    while(!pq.empty()){
        // get the current node
        int u = pq.top().second;
        pq.pop();

        // loop through the neighbors of the current node
        for(auto& neighbor : adj_list[u]){
            int v = neighbor.first;
            int w = neighbor.second;

            // if the distance to v is shorter by going through u
            if(dist[v] > dist[u] + w){
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}

/*
// the required signature is fixed as follows
void dijkstra_np(void *out, const void** in){
    const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
    // size * size adjacency table
    const std::int32_t **adj_table = reinterpret_cast<const std::int32_t **>(in[1]);
    // the start node
    const std::int32_t start = *reinterpret_cast<const std::int32_t *>(in[2]);
    // output a list of distances
    std::int32_t *dist = reinterpret_cast<std::int32_t *>(out);

    // initialize the distance vector
    int[] dist(size, INT_MAX);
    dist[start] = 0;

    // initialize the priority queue
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
}
 */

// Expose `dijkstra` as a function in the module min_dist
PYBIND11_MODULE(_min_dist, m) {
    m.doc() = "dijkstra using stl";
    m.def("dijkstra_stl", &dijkstra_stl, "dijkstra using stl");
    //m.def("dijkstra_np", &dijkstra, "dijkstra using numpy array")
}