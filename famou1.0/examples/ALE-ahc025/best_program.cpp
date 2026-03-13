#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <limits>

// --- Utilities ---
struct Timer {
    std::chrono::high_resolution_clock::time_point start_time;
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    double get_time() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
    }
};

class Solver {
public:
    int N, D, Q;
    int queries_used = 0;
    std::vector<int> d_out;
    Timer timer;
    std::mt19937 rng;

    const double TOTAL_TIME_LIMIT = 1.90; 

    Solver() {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(NULL);
        std::cin >> N >> D >> Q;
        d_out.assign(N, 0);
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    void run() {
        if (should_use_high_q_strategy()) {
            high_q_strategy();
        } else {
            low_q_strategy();
        }

        while (queries_used < Q) {
            perform_query({0}, {1}); 
        }

        for (int i = 0; i < N; ++i) {
            std::cout << d_out[i] << (i == N - 1 ? "" : " ");
        }
        std::cout << std::endl;
    }

private:
    char perform_query(const std::vector<int>& L, const std::vector<int>& R) {
        if (queries_used >= Q) return '='; 
        
        queries_used++;
        std::cout << L.size() << " " << R.size();
        for (int item : L) std::cout << " " << item;
        for (int item : R) std::cout << " " << item;
        std::cout << std::endl; 
        
        char result;
        std::cin >> result;
        return result;
    }

    char compare_items(int u, int v) {
        if (u == v) return '=';
        if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return '=';
        return perform_query({u}, {v});
    }

    char compare_groups(const std::vector<int>& group1, const std::vector<int>& group2) {
        if (group1.empty() && group2.empty()) return '=';
        if (group1.empty()) return '<';
        if (group2.empty()) return '>';
        if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return '=';
        
        return perform_query(group1, group2);
    }

    void insertion_sort_items(std::vector<int>& items) {
        if (items.empty()) return;
        std::vector<int> sorted_items;
        sorted_items.reserve(items.size());
        sorted_items.push_back(items[0]);

        for (size_t i = 1; i < items.size(); ++i) {
            int item = items[i];
            if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) {
                sorted_items.push_back(item); 
                continue;
            }
            auto it = std::lower_bound(sorted_items.begin(), sorted_items.end(), item, 
                [&](int a, int b){
                    if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return false; 
                    return compare_items(a, b) == '<';
                });
            sorted_items.insert(it, item);
        }
        items = sorted_items;
    }

    bool should_use_high_q_strategy() {
        std::vector<long long> log_sum(N + 1, 0);
        for (int i = 1; i <= N; ++i) {
            log_sum[i] = log_sum[i - 1] + (i > 1 ? static_cast<long long>(ceil(log2(i))) : 0);
        }
        
        long long full_sort_cost = log_sum[N];

        long long partition_cost = 0;
        if (D > 1) {
            long long initial_group_sort_cost = log_sum[D];
            long long greedy_assignment_cost = (long long)(N - D) * static_cast<long long>(ceil(log2(D)));
            partition_cost = initial_group_sort_cost + greedy_assignment_cost;
        }
        
        long long total_estimated_cost = full_sort_cost + partition_cost;
        
        return Q >= total_estimated_cost + N;
    }

    void high_q_strategy() {
        std::vector<int> sorted_items(N);
        std::iota(sorted_items.begin(), sorted_items.end(), 0);
        insertion_sort_items(sorted_items);
        std::reverse(sorted_items.begin(), sorted_items.end());

        std::vector<std::vector<int>> groups(D);
        std::vector<int> group_order_indices(D); 
        std::iota(group_order_indices.begin(), group_order_indices.end(), 0);

        for (int i = 0; i < std::min(N, D); ++i) {
            groups[i].push_back(sorted_items[i]);
        }

        if (D > 1) {
            std::sort(group_order_indices.begin(), group_order_indices.end(), 
                [&](int a_idx, int b_idx){
                    if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return a_idx < b_idx;
                    return compare_groups(groups[a_idx], groups[b_idx]) == '<';
                });
        }
        
        for (int i = D; i < N; ++i) {
            if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) {
                groups[i % D].push_back(sorted_items[i]);
                continue;
            }
            int item = sorted_items[i];
            
            int lightest_group_g_idx = group_order_indices[0];
            groups[lightest_group_g_idx].push_back(item);

            group_order_indices.erase(group_order_indices.begin()); 
            
            auto it = std::lower_bound(group_order_indices.begin(), group_order_indices.end(), lightest_group_g_idx,
                [&](int a_idx, int b_idx){
                    if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return false;
                    return compare_groups(groups[a_idx], groups[b_idx]) == '<';
                });
            group_order_indices.insert(it, lightest_group_g_idx);
        }
        
        for(int i = 0; i < D; ++i) {
            for(int item : groups[i]) {
                d_out[item] = i;
            }
        }
    }

    void low_q_strategy() {
        long long refine_budget = std::max((long long)D*D, (long long)Q / 4);
        long long info_budget = Q - refine_budget;

        std::vector<double> pseudo_weights = estimate_pseudo_weights(info_budget);
        
        std::vector<std::vector<int>> groups(D);
        std::vector<double> group_pseudo_sums(D, 0.0);
        initial_partition(pseudo_weights, groups, group_pseudo_sums);
        
        double time_after_info = timer.get_time();
        double remaining_time = TOTAL_TIME_LIMIT - time_after_info;
        
        double sa_time_budget = remaining_time * 0.6;
        double sa_end_time = time_after_info + sa_time_budget;

        run_simulated_annealing(pseudo_weights, groups, group_pseudo_sums, sa_end_time);
        run_query_based_refinement(pseudo_weights, groups, group_pseudo_sums);

        for(int i = 0; i < D; ++i) {
            for(int item : groups[i]) {
                d_out[item] = i;
            }
        }
    }

    std::vector<double> estimate_pseudo_weights(long long info_budget) {
        std::vector<long long> log_sum(N + 1, 0);
        for (int i = 1; i <= N; ++i) log_sum[i] = log_sum[i - 1] + (i > 1 ? (long long)ceil(log2(i)) : 0);
        
        auto cost_func = [&](int p) {
            if (p <= 0) return (long long)Q + 1;
            long long calibration_cost = (p >= 4) ? (long long)ceil(log2(p)) : 0;
            return log_sum[p] + (long long)(N - p) * (long long)ceil(log2(p + 1.0)) + calibration_cost;
        };
        
        int p_opt = 0;
        for (int p = 1; p <= N; ++p) {
            if (cost_func(p) <= info_budget) p_opt = p; else break;
        }
        
        std::vector<int> item_indices(N);
        std::iota(item_indices.begin(), item_indices.end(), 0);
        std::shuffle(item_indices.begin(), item_indices.end(), rng);

        std::vector<int> pivots;
        pivots.reserve(p_opt);
        for (int i = 0; i < p_opt; ++i) pivots.push_back(item_indices[i]);
        insertion_sort_items(pivots);

        double base = exp(6.0 / (std::max(1, 2 * p_opt - 1)));
        if (pivots.size() >= 4 && Q - queries_used > log2(pivots.size())) {
            int k = pivots.size();
            int heaviest_pivot = pivots.back();
            int low = 1, high = k - 1, m = 0;
            while (low <= high) {
                if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) break;
                int mid = low + (high - low) / 2;
                if (mid == 0) { low = mid + 1; continue; }
                std::vector<int> left_set;
                for (int i = 0; i < mid; ++i) left_set.push_back(pivots[i]);
                if (compare_groups(left_set, {heaviest_pivot}) == '<') {
                    m = mid; low = mid + 1;
                } else high = mid - 1;
            }
            if (m > 0) {
                double B_low = 1.0, B_high = 3.0;
                for (int iter = 0; iter < 30; ++iter) {
                    double B_mid = (B_low + B_high) / 2.0;
                    if (B_mid <= 1.0) { B_low = 1.000001; continue; }
                    double val = pow(B_mid, k - 1) * (B_mid - 1) - (pow(B_mid, m) - 1);
                    if (std::isinf(val) || val > 0) B_high = B_mid; else B_low = B_mid;
                }
                if (B_low > 1.000001) base = B_low;
            }
        }

        std::vector<double> pseudo_weights(N);
        for (size_t i = 0; i < pivots.size(); ++i) {
            pseudo_weights[pivots[i]] = pow(base, (double)i);
        }
        for (int i = p_opt; i < N; ++i) {
            int item = item_indices[i];
            if (pivots.empty() || queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) {
                pseudo_weights[item] = 1.0;
                continue;
            }
            auto it = std::lower_bound(pivots.begin(), pivots.end(), item,
                [&](int a, int b){
                    if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return false;
                    return compare_items(a, b) == '<';
                });
            int bucket_idx = std::distance(pivots.begin(), it);
            double rank = (double)bucket_idx - 0.5;
            pseudo_weights[item] = pow(base, rank);
        }
        
        double total_pw = std::accumulate(pseudo_weights.begin(), pseudo_weights.end(), 0.0);
        if (total_pw > 1e-9) {
            double avg_pw = total_pw / N;
            if (avg_pw > 1e-9) {
                for(int i=0; i<N; ++i) pseudo_weights[i] /= avg_pw;
            }
        }
        return pseudo_weights;
    }

    void initial_partition(const std::vector<double>& pseudo_weights, std::vector<std::vector<int>>& groups, std::vector<double>& group_pseudo_sums) {
        std::vector<std::pair<double, int>> weighted_items;
        weighted_items.reserve(N);
        for(int i=0; i<N; ++i) weighted_items.push_back({-pseudo_weights[i], i});
        std::sort(weighted_items.begin(), weighted_items.end());

        for (int i = 0; i < N; ++i) {
            int item_idx = weighted_items[i].second;
            auto min_it = std::min_element(group_pseudo_sums.begin(), group_pseudo_sums.end());
            int target_group = std::distance(group_pseudo_sums.begin(), min_it);
            groups[target_group].push_back(item_idx);
            group_pseudo_sums[target_group] += pseudo_weights[item_idx];
        }
    }

    void run_simulated_annealing(const std::vector<double>& pseudo_weights, std::vector<std::vector<int>>& groups, std::vector<double>& group_pseudo_sums, double sa_end_time) {
        std::vector<int> d_sa(N);
        for(int g_idx = 0; g_idx < D; ++g_idx) {
            for(int item : groups[g_idx]) d_sa[item] = g_idx;
        }
        
        std::vector<std::pair<double, int>> weighted_items;
        for(int i=0; i<N; ++i) weighted_items.push_back({-pseudo_weights[i], i});
        std::sort(weighted_items.begin(), weighted_items.end());

        auto select_item_smart = [&]() -> int {
            if (std::uniform_real_distribution<>(0,1)(rng) < 0.5 && !weighted_items.empty()) {
                int top_k = std::max(1, N / 4);
                int idx = rng() % top_k;
                return weighted_items[idx].second;
            } else {
                return rng() % N;
            }
        };

        double start_temp = 2.0 * N / D, end_temp = 0.01;
        double start_time_sa = timer.get_time();

        while(timer.get_time() < sa_end_time) {
            double progress = (timer.get_time() - start_time_sa) / std::max(1e-9, sa_end_time - start_time_sa);
            if (progress >= 1.0) break;
            double temp = start_temp * pow(end_temp / start_temp, progress);
            
            if (rng() % 4 != 0 || D < 2) { 
                int item_to_move = select_item_smart();
                int old_group = d_sa[item_to_move];
                int new_group = rng() % D;
                if (old_group == new_group || (groups[old_group].size() <= 1 && D > 1)) continue;
                
                double w = pseudo_weights[item_to_move];
                double s_old = group_pseudo_sums[old_group], s_new = group_pseudo_sums[new_group];
                double score_delta = pow(s_old - w, 2) + pow(s_new + w, 2) - (pow(s_old, 2) + pow(s_new, 2));

                if (score_delta < 0 || std::uniform_real_distribution<>(0, 1)(rng) < exp(-score_delta / temp)) {
                    group_pseudo_sums[old_group] -= w;
                    group_pseudo_sums[new_group] += w;
                    d_sa[item_to_move] = new_group;
                    auto& g_old = groups[old_group];
                    g_old.erase(std::remove(g_old.begin(), g_old.end(), item_to_move), g_old.end());
                    groups[new_group].push_back(item_to_move);
                }
            } else { 
                int item1 = select_item_smart(), item2 = select_item_smart();
                if (item1 == item2) continue;
                int g1 = d_sa[item1], g2 = d_sa[item2];
                if (g1 == g2) continue;

                double w1 = pseudo_weights[item1], w2 = pseudo_weights[item2];
                double s1 = group_pseudo_sums[g1], s2 = group_pseudo_sums[g2];
                double next_s1 = s1 - w1 + w2, next_s2 = s2 - w2 + w1;
                double score_delta = pow(next_s1, 2) + pow(next_s2, 2) - (pow(s1, 2) + pow(s2, 2));

                if (score_delta < 0 || std::uniform_real_distribution<>(0, 1)(rng) < exp(-score_delta / temp)) {
                    group_pseudo_sums[g1] = next_s1; group_pseudo_sums[g2] = next_s2;
                    std::swap(d_sa[item1], d_sa[item2]);
                    auto& g1_list = groups[g1]; g1_list.erase(std::remove(g1_list.begin(), g1_list.end(), item1), g1_list.end()); groups[g2].push_back(item1);
                    auto& g2_list = groups[g2]; g2_list.erase(std::remove(g2_list.begin(), g2_list.end(), item2), g2_list.end()); groups[g1].push_back(item2);
                }
            }
        }
        for(auto& g : groups) g.clear();
        for(int i=0; i<N; ++i) groups[d_sa[i]].push_back(i);
    }
    
    void run_query_based_refinement(const std::vector<double>& pseudo_weights, std::vector<std::vector<int>>& groups, std::vector<double>& group_pseudo_sums) {
        if (D > 1 && Q - queries_used > D * log2(D) + D) {
            systematic_refinement(pseudo_weights, groups, group_pseudo_sums);
        } else {
            opportunistic_refinement(pseudo_weights, groups, group_pseudo_sums);
        }
    }

    void systematic_refinement(const std::vector<double>& pseudo_weights, std::vector<std::vector<int>>& groups, std::vector<double>& group_pseudo_sums) {
        std::vector<int> sorted_groups(D);
        std::iota(sorted_groups.begin(), sorted_groups.end(), 0);
        std::sort(sorted_groups.begin(), sorted_groups.end(), [&](int a, int b) {
            if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return a < b;
            return compare_groups(groups[a], groups[b]) == '<';
        });

        for(int iter = 0; iter < 2 * D * D && queries_used < Q && timer.get_time() < TOTAL_TIME_LIMIT; ++iter) {
            if (sorted_groups.size() < 2) break;
            int g_h_idx = sorted_groups.back(), g_l_idx = sorted_groups.front();
            if (g_h_idx == g_l_idx || groups[g_h_idx].empty()) break;
            
            if (compare_groups(groups[g_h_idx], groups[g_l_idx]) != '>') break;
            if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) break;

            bool action_taken = false;
            if (try_swap_between_groups(g_h_idx, g_l_idx, pseudo_weights, groups, group_pseudo_sums)) {
                action_taken = true;
            } else if (groups[g_h_idx].size() > 1) {
                move_item_between_groups(g_h_idx, g_l_idx, pseudo_weights, groups, group_pseudo_sums);
                action_taken = true;
            }

            if (action_taken) {
                update_group_position_in_sorted_list(g_h_idx, sorted_groups, groups);
                update_group_position_in_sorted_list(g_l_idx, sorted_groups, groups);
            } else break;
        }
    }

    bool try_swap_between_groups(int g1_idx, int g2_idx, const std::vector<double>& pseudo_weights, std::vector<std::vector<int>>& groups, std::vector<double>& group_pseudo_sums) {
        if (groups[g1_idx].empty() || groups[g2_idx].empty() || queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return false;

        int item1 = *std::min_element(groups[g1_idx].begin(), groups[g1_idx].end(), [&](int a, int b){ return pseudo_weights[a] < pseudo_weights[b]; });
        int item2 = *std::max_element(groups[g2_idx].begin(), groups[g2_idx].end(), [&](int a, int b){ return pseudo_weights[a] < pseudo_weights[b]; });

        if (compare_items(item1, item2) == '>') {
            auto& g1 = groups[g1_idx]; auto& g2 = groups[g2_idx];
            g1.erase(std::remove(g1.begin(), g1.end(), item1), g1.end()); g2.erase(std::remove(g2.begin(), g2.end(), item2), g2.end());
            g1.push_back(item2); g2.push_back(item1);
            group_pseudo_sums[g1_idx] += pseudo_weights[item2] - pseudo_weights[item1];
            group_pseudo_sums[g2_idx] += pseudo_weights[item1] - pseudo_weights[item2];
            return true;
        }
        return false;
    }
    
    void move_item_between_groups(int from_idx, int to_idx, const std::vector<double>& pseudo_weights, std::vector<std::vector<int>>& groups, std::vector<double>& group_pseudo_sums) {
        if (groups[from_idx].empty()) return;
        int item_to_move = *std::min_element(groups[from_idx].begin(), groups[from_idx].end(), [&](int a, int b){ return pseudo_weights[a] < pseudo_weights[b]; });
        auto& group_from = groups[from_idx];
        group_from.erase(std::remove(group_from.begin(), group_from.end(), item_to_move), group_from.end());
        groups[to_idx].push_back(item_to_move);
        group_pseudo_sums[from_idx] -= pseudo_weights[item_to_move];
        group_pseudo_sums[to_idx] += pseudo_weights[item_to_move];
    }

    void update_group_position_in_sorted_list(int g_idx, std::vector<int>& sorted_groups, const std::vector<std::vector<int>>& groups) {
        auto it_pos = std::find(sorted_groups.begin(), sorted_groups.end(), g_idx);
        if (it_pos != sorted_groups.end()) sorted_groups.erase(it_pos);
        if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) {
            sorted_groups.push_back(g_idx); return;
        }
        auto it_insert = std::lower_bound(sorted_groups.begin(), sorted_groups.end(), g_idx,
            [&](int a_idx, int b_idx){
                if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT) return false;
                return compare_groups(groups[a_idx], groups[b_idx]) == '<';
            });
        sorted_groups.insert(it_insert, g_idx);
    }
    
    void opportunistic_refinement(const std::vector<double>& pseudo_weights, std::vector<std::vector<int>>& groups, std::vector<double>& group_pseudo_sums) {
        while (queries_used < Q && timer.get_time() < TOTAL_TIME_LIMIT) {
            if (D <= 1) break;
            int g1 = std::distance(group_pseudo_sums.begin(), std::max_element(group_pseudo_sums.begin(), group_pseudo_sums.end()));
            int g2 = std::distance(group_pseudo_sums.begin(), std::min_element(group_pseudo_sums.begin(), group_pseudo_sums.end()));
            if (g1 == g2) { g1 = rng() % D; do { g2 = rng() % D; } while (g1 == g2); }
            if (groups[g1].empty() || groups[g2].empty()) continue;

            char res = compare_groups(groups[g1], groups[g2]);
            if (queries_used >= Q || timer.get_time() >= TOTAL_TIME_LIMIT || res == '=') break;
            
            int heavier = (res == '>') ? g1 : g2, lighter = (res == '>') ? g2 : g1;
            if (!try_swap_between_groups(heavier, lighter, pseudo_weights, groups, group_pseudo_sums)) {
                if (groups[heavier].size() > 1) { 
                    move_item_between_groups(heavier, lighter, pseudo_weights, groups, group_pseudo_sums);
                }
            }
        }
    }
};

int main() {
    Solver solver;
    solver.run();
    return 0;
}