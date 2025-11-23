#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <queue>
#include <cmath>
#include <random>

using namespace std;

// Timer
auto start_time = chrono::steady_clock::now();
double time_limit = 1.90; // More conservative time limit

// Constants
const int MAX_L = 100000;
const int DR[] = {-1, 1, 0, 0}; // U, D
const int DC[] = {0, 0, -1, 1}; // L, R
const char D_CHAR[] = {'U', 'D', 'L', 'R'};

// Globals
int N;
vector<string> h_walls, v_walls;
vector<vector<int>> d_grid;
vector<int> d_flat;
vector<vector<int>> adj;
vector<int> dist_flat;
vector<int> pred_flat;
vector<pair<int, int>> coords;
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// Utilities
inline int to_id(int r, int c) { return r * N + c; }

// Input and Graph Building
void read_input() {
    cin >> N;
    h_walls.resize(N - 1);
    for (int i = 0; i < N - 1; ++i) cin >> h_walls[i];
    v_walls.resize(N);
    for (int i = 0; i < N; ++i) cin >> v_walls[i];
    d_grid.assign(N, vector<int>(N));
    d_flat.resize(N * N);
    coords.resize(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> d_grid[i][j];
            int id = to_id(i, j);
            d_flat[id] = d_grid[i][j];
            coords[id] = {i, j};
        }
    }
}

void build_graph() {
    adj.assign(N * N, vector<int>());
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int u = to_id(r, c);
            for (int i = 0; i < 4; ++i) {
                int nr = r + DR[i], nc = c + DC[i];
                if (nr >= 0 && nr < N && nc >= 0 && nc < N) {
                    bool is_wall = (DR[i] == 0) ? (v_walls[r][min(c, nc)] == '1') : (h_walls[min(r, nr)][c] == '1');
                    if (!is_wall) {
                        adj[u].push_back(to_id(nr, nc));
                    }
                }
            }
        }
    }
}

void compute_apsp() {
    int V = N * N;
    dist_flat.assign(V * V, -1);
    pred_flat.assign(V * V, -1);
    for (int start_node = 0; start_node < V; ++start_node) {
        queue<int> q;
        q.push(start_node);
        dist_flat[start_node * V + start_node] = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (dist_flat[start_node * V + v] == -1) {
                    dist_flat[start_node * V + v] = dist_flat[start_node * V + u] + 1;
                    pred_flat[start_node * V + v] = u;
                    q.push(v);
                }
            }
        }
    }
}

// Path and Score Calculation
string reconstruct_path_string(const vector<int>& key_nodes) {
    string path_str = "";
    path_str.reserve(MAX_L);
    int V = N * N;
    for (size_t i = 0; i < key_nodes.size() - 1; ++i) {
        int start_node = key_nodes[i];
        int end_node = key_nodes[i + 1];
        if (start_node == end_node) continue;
        
        string segment_moves = "";
        int curr = end_node;
        while (curr != start_node) {
            int prev = pred_flat[start_node * V + curr];
            auto p_coord = coords[prev];
            auto c_coord = coords[curr];
            for (int dir = 0; dir < 4; ++dir) {
                if (p_coord.first + DR[dir] == c_coord.first && p_coord.second + DC[dir] == c_coord.second) {
                    segment_moves += D_CHAR[dir];
                    break;
                }
            }
            curr = prev;
        }
        reverse(segment_moves.begin(), segment_moves.end());
        path_str += segment_moves;
    }
    return path_str;
}

pair<double, long long> calculate_score_and_len(const vector<int>& key_nodes, vector<bool>& visited_once) {
    long long total_len = 0;
    vector<vector<long long>> visits(N * N);
    fill(visited_once.begin(), visited_once.end(), false);
    int V = N * N;

    int start_node_for_path = key_nodes[0];
    visits[start_node_for_path].push_back(0);
    visited_once[start_node_for_path] = true;

    for (size_t i = 0; i < key_nodes.size() - 1; ++i) {
        int start_node = key_nodes[i];
        int end_node = key_nodes[i+1];
        
        long long path_len = dist_flat[start_node * V + end_node];
        if (path_len == -1) return {1e18, -1};
        
        if (path_len > 0) {
            int curr = end_node;
            vector<int> path_segment;
            path_segment.reserve(path_len);
            while(curr != start_node) {
                path_segment.push_back(curr);
                curr = pred_flat[start_node * V + curr];
            }
            reverse(path_segment.begin(), path_segment.end());
            
            for(size_t j = 0; j < path_segment.size(); ++j) {
                int node = path_segment[j];
                visits[node].push_back(total_len + j + 1);
                visited_once[node] = true;
            }
        }
        total_len += path_len;
    }

    if (total_len > MAX_L) return {1e18, total_len};
    
    for (int i = 0; i < V; ++i) {
        if (!visited_once[i]) return {1e18, total_len};
    }

    if (total_len == 0) return {0.0, 0};

    double total_dirtiness_integral = 0;
    for (int i = 0; i < V; ++i) {
        if(visits[i].empty()) continue;
        long long last_visit_time = visits[i].back() - total_len;
        double node_dirt_integral = 0;
        for (long long visit_time : visits[i]) {
            long long delta = visit_time - last_visit_time;
            node_dirt_integral += (double)delta * (delta - 1.0) / 2.0;
            last_visit_time = visit_time;
        }
        total_dirtiness_integral += node_dirt_integral * d_flat[i];
    }
    
    return {total_dirtiness_integral / total_len, total_len};
}

// Initial Tour Generation
string initial_path_moves;
vector<bool> visited_dfs;
void generate_initial_tour_dfs(int r, int c) {
    visited_dfs[to_id(r, c)] = true;
    char D_CHAR_REV[] = {'D', 'U', 'R', 'L'};
    for (int i = 0; i < 4; ++i) {
        int nr = r + DR[i], nc = c + DC[i];
        if (nr >= 0 && nr < N && nc >= 0 && nc < N && !visited_dfs[to_id(nr, nc)]) {
            bool is_wall = (DR[i] == 0) ? (v_walls[r][min(c, nc)] == '1') : (h_walls[min(r, nr)][c] == '1');
            if (!is_wall) {
                initial_path_moves += D_CHAR[i];
                generate_initial_tour_dfs(nr, nc);
                initial_path_moves += D_CHAR_REV[i];
            }
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    read_input();
    build_graph();
    compute_apsp();

    // Phase 1: Greedy Construction
    visited_dfs.assign(N * N, false);
    generate_initial_tour_dfs(0, 0);

    vector<long long> last_visit(N * N, 0);
    long long current_time = 0;
    int current_pos = 0;
    
    {
        int r = 0, c = 0;
        last_visit[to_id(r, c)] = 0;
        for (char move : initial_path_moves) {
            current_time++;
            if (move == 'U') r--; else if (move == 'D') r++; else if (move == 'L') c--; else if (move == 'R') c++;
            last_visit[to_id(r, c)] = current_time;
        }
        current_pos = to_id(r,c);
    }
    
    vector<int> key_nodes;
    key_nodes.push_back(0);
    long long constructed_path_len = 0;

    double construction_time_limit = 1.0;
    int V = N * N;

    while (chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - start_time).count() < construction_time_limit) {
        int w_star = -1;
        double max_score = -1.0;

        for (int w = 0; w < V; ++w) {
            long long travel_cost = dist_flat[current_pos * V + w];
            if (travel_cost <= 0) continue;

            long long time_since_last_visit = current_time - last_visit[w];
            double effective_time_at_arrival = (double)time_since_last_visit + travel_cost;
            double reward = (double)d_flat[w] * effective_time_at_arrival * effective_time_at_arrival;
            double score = reward / (double)travel_cost;

            if (score > max_score) {
                max_score = score;
                w_star = w;
            }
        }

        if (w_star == -1) break;
        
        long long path_len_to_w = dist_flat[current_pos * V + w_star];
        if (constructed_path_len + path_len_to_w + dist_flat[w_star * V + 0] > MAX_L) break;

        vector<int> path_rev;
        int curr = w_star;
        while(curr != current_pos) { path_rev.push_back(curr); curr = pred_flat[current_pos * V + curr]; }
        reverse(path_rev.begin(), path_rev.end());
        for(int node : path_rev) { current_time++; last_visit[node] = current_time; }
        
        constructed_path_len += path_len_to_w;
        current_pos = w_star;
        key_nodes.push_back(w_star);
    }
    
    vector<bool> visited_once(V);
    calculate_score_and_len(key_nodes, visited_once);
    vector<int> unvisited_nodes;
    for(int i=0; i<V; ++i) if(!visited_once[i]) unvisited_nodes.push_back(i);

    for(int node_to_insert : unvisited_nodes){
        long long best_cost_increase = -1;
        int best_insert_pos = -1;
        for(size_t i=0; i < key_nodes.size() - 1; ++i){
            long long cost_increase = dist_flat[key_nodes[i]*V + node_to_insert] + dist_flat[node_to_insert*V + key_nodes[i+1]] - dist_flat[key_nodes[i]*V + key_nodes[i+1]];
            if(best_insert_pos == -1 || cost_increase < best_cost_increase){
                best_cost_increase = cost_increase;
                best_insert_pos = i + 1;
            }
        }
        key_nodes.insert(key_nodes.begin() + best_insert_pos, node_to_insert);
    }
    key_nodes.push_back(0);

    // Phase 2: Simulated Annealing
    auto current_eval = calculate_score_and_len(key_nodes, visited_once);
    double current_score = current_eval.first;
    
    vector<int> best_key_nodes = key_nodes;
    double best_score = current_score;

    double T_start = best_score > 0 ? best_score * 0.05 : 1000.0;
    double T_end = 1e-4;
    int iter = 0;
    
    while (true) {
        iter++;
        double elapsed = chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now() - start_time).count();
        if (iter % 100 == 0) {
            if (elapsed > time_limit) break;
        }

        vector<int> next_key_nodes = key_nodes;
        if (next_key_nodes.size() <= 3) break;

        int i = uniform_int_distribution<int>(1, next_key_nodes.size() - 2)(rng);
        int j = uniform_int_distribution<int>(1, next_key_nodes.size() - 2)(rng);
        if (i == j) continue;

        if (rng() % 2 == 0) {
            if (i > j) swap(i, j);
            reverse(next_key_nodes.begin() + i, next_key_nodes.begin() + j + 1);
        } else {
            int node_to_move = next_key_nodes[i];
            next_key_nodes.erase(next_key_nodes.begin() + i);
            next_key_nodes.insert(next_key_nodes.begin() + j, node_to_move);
        }

        auto next_eval = calculate_score_and_len(next_key_nodes, visited_once);
        double next_score = next_eval.first;
        
        double temp = T_start * pow(T_end / T_start, elapsed / time_limit);
        
        if (next_score < current_score || (temp > 1e-9 && uniform_real_distribution<double>(0.0, 1.0)(rng) < exp((current_score - next_score) / temp))) {
            key_nodes = next_key_nodes;
            current_score = next_score;
            if (current_score < best_score) {
                best_score = current_score;
                best_key_nodes = key_nodes;
            }
        }
    }

    // Phase 3: Output
    cout << reconstruct_path_string(best_key_nodes) << endl;

    return 0;
}