"""
A high-performance program for constructing optimal 16-point configurations
in 2D space by minimizing the ratio of maximum to minimum pairwise distance.

This program solves a smooth, constrained optimization problem reformulated from the
original non-smooth ratio objective. It minimizes the maximum squared distance,
subject to the constraint that the minimum squared distance is at least 1.

Key improvements include:
- A fully vectorized Jacobian for the constraint function, providing a significant
  performance boost over iterative calculations.
- Correction of a variable scope bug present in the previous version.
- An expanded multi-start strategy with an additional initial configuration to
  more robustly explore the solution space.
- Increased optimizer iterations to achieve higher precision.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from itertools import combinations

def calculate_ratio(points):
    """Calculate the ratio between maximum and minimum pairwise distances."""
    if points is None or len(points) < 2:
        return float('inf')
    
    distances = pdist(points)
    
    if len(distances) == 0:
        return float('inf')
    
    d_max = np.max(distances)
    d_min = np.min(distances)
    
    if d_min < 1e-9:  # Treat very small distances as zero to avoid instability
        return float('inf')
    
    return d_max / d_min

def construct_16_points():
    """
    Construct 16 points in 2D space to minimize the ratio d_max/d_min
    using a multi-start constrained optimization approach (SLSQP) with a
    highly efficient vectorized Jacobian.
    """
    N_POINTS = 16
    N_VARS = N_POINTS * 2
    
    best_points = None
    best_ratio_sq = float('inf')

    # --- Pre-compute indices for vectorization ---
    # This is done once to speed up the Jacobian calculation inside the solver loop.
    indices = list(combinations(range(N_POINTS), 2))
    N_PAIRS = len(indices)
    I = np.array([i for i, j in indices])
    J = np.array([j for i, j in indices])

    # --- Initial Configurations ---
    # A diverse set of starting points is crucial for finding a good global minimum.

    # Config 1: Hexagonal Lattice Section
    points_hex = []
    sqrt3_div_2 = np.sqrt(3) / 2.0
    for v_idx in range(4):
        for u_idx in range(4):
            x = (u_idx - 1.5) + 0.5 * (v_idx - 1.5)
            y = (v_idx - 1.5) * sqrt3_div_2
            points_hex.append([x, y])
    initial_hex = np.array(points_hex)

    # Config 2: 1-5-10 Concentric Ring Structure (known to be near-optimal)
    points_1_5_10 = [[0, 0]]
    r1 = 1.0
    for i in range(5):
        angle = i * 2 * np.pi / 5
        points_1_5_10.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 1.992  # Fine-tuned radius based on known good solutions
    initial_rotation = np.pi / 10
    for i in range(10):
        angle = i * 2 * np.pi / 10 + initial_rotation
        points_1_5_10.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    initial_1_5_10 = np.array(points_1_5_10)
    
    # Config 3: Two-Ring Structure (6-10)
    points_6_10 = []
    for i in range(6):
        angle = i * np.pi / 3
        points_6_10.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    for i in range(10):
        angle = i * 2 * np.pi / 10 + np.pi/10
        points_6_10.append([1.9 * np.cos(angle), 1.9 * np.sin(angle)])
    initial_6_10 = np.array(points_6_10)

    # Config 4: 4x4 Grid
    points_grid = []
    for i in range(4):
        for j in range(4):
            points_grid.append([i - 1.5, j - 1.5])
    initial_grid = np.array(points_grid)

    # Config 5: Random Start
    np.random.seed(42)
    initial_random = np.random.rand(N_POINTS, 2) * 5 - 2.5

    initial_configs = {
        "hexagonal": initial_hex,
        "concentric_1_5_10": initial_1_5_10,
        "concentric_6_10": initial_6_10,
        "grid_4x4": initial_grid,
        "random": initial_random,
    }

    # --- Setup for SLSQP Optimization ---
    # The optimization variable `x` is a flat array: [x1, y1, ..., x16, y16, D_max_sq]
    
    objective_func = lambda x: x[-1]
    
    def objective_jac(x):
        grad = np.zeros_like(x)
        grad[-1] = 1.0
        return grad

    # --- Vectorized Constraint and Jacobian Functions ---
    # Defined within this scope to have access to N_VARS, N_POINTS, etc.
    def constraints_func(x):
        points = x[:N_VARS].reshape(N_POINTS, 2)
        D_max_sq = x[-1]
        sq_dists = pdist(points, 'sqeuclidean')
        # c1: d_ij^2 >= 1  =>  d_ij^2 - 1 >= 0
        c1 = sq_dists - 1.0
        # c2: d_ij^2 <= D_max_sq  =>  D_max_sq - d_ij^2 >= 0
        c2 = D_max_sq - sq_dists
        return np.concatenate((c1, c2))

    def constraints_jac(x):
        points = x[:N_VARS].reshape(N_POINTS, 2)
        jac = np.zeros((2 * N_PAIRS, N_VARS + 1))
        
        # Calculate all 2*(pi - pj) vectors in a single operation
        diffs = 2 * (points[I] - points[J])
        
        # Row indices for the first block of constraints
        k = np.arange(N_PAIRS)
        
        # Populate Jacobian for c1 constraints (d_ij^2 - 1) using vectorized assignment
        jac[k, 2 * I] = diffs[:, 0]
        jac[k, 2 * I + 1] = diffs[:, 1]
        jac[k, 2 * J] = -diffs[:, 0]
        jac[k, 2 * J + 1] = -diffs[:, 1]
        
        # Populate Jacobian for c2 constraints (D_max_sq - d_ij^2)
        jac[k + N_PAIRS, 2 * I] = -diffs[:, 0]
        jac[k + N_PAIRS, 2 * I + 1] = -diffs[:, 1]
        jac[k + N_PAIRS, 2 * J] = diffs[:, 0]
        jac[k + N_PAIRS, 2 * J + 1] = diffs[:, 1]
        
        # Derivative of c2 with respect to D_max_sq is 1
        jac[N_PAIRS:, -1] = 1.0
        return jac

    cons = {'type': 'ineq', 'fun': constraints_func, 'jac': constraints_jac}
    optimizer_options = {'maxiter': 3000, 'ftol': 1e-12, 'disp': False}

    for name, config in initial_configs.items():
        initial_points = config.copy()
        dists = pdist(initial_points)
        min_dist = np.min(dists)
        if min_dist > 1e-9:
             initial_points /= min_dist
        
        initial_d_max_sq = np.max(pdist(initial_points)**2)
        x0 = np.append(initial_points.flatten(), initial_d_max_sq)
        
        result = minimize(
            objective_func,
            x0,
            method='SLSQP',
            jac=objective_jac,
            constraints=cons,
            options=optimizer_options
        )
        
        if result.success:
            current_ratio_sq = result.fun
            if current_ratio_sq < best_ratio_sq:
                best_ratio_sq = current_ratio_sq
                best_points = result.x[:N_VARS].reshape(N_POINTS, 2)
    
    if best_points is None:
        best_points = initial_1_5_10
    
    return best_points

def run_construction():
    """Main function that runs the construction and returns results."""
    try:
        points = construct_16_points()
        if points is not None:
            # Center and normalize the final configuration for a canonical representation
            points -= np.mean(points, axis=0)
            min_dist = np.min(pdist(points))
            if min_dist > 1e-9:
                points /= min_dist
        return points
    except Exception as e:
        print(f"An error occurred during construction: {e}")
        return None

if __name__ == "__main__":
    points = run_construction()
    
    if points is not None:
        ratio = calculate_ratio(points)
        ratio_squared = ratio**2
        
        print(f"Achieved ratio squared: {ratio_squared:.20f}")
        target_sq = 12.889266112
        print(f"Target ratio squared:   {target_sq:.20f} (ratio ≈ {np.sqrt(target_sq):.20f})")
        if ratio_squared < target_sq:
            print("\nSuccess: Target beaten!")
        else:
            print("\nFailure: Target not beaten.")
        
    else:
        print("Construction failed.")

    #@title Construction verification
    import scipy as sp

    if points is not None:
        # print(points)
        print(f'\nConstruction has {len(points)} points in {points.shape[1]} dimensions.')
        pairwise_distances = sp.spatial.distance.pdist(points)
        min_distance = np.min(pairwise_distances)
        max_distance = np.max(pairwise_distances)

        final_ratio_squared = (max_distance / min_distance)**2
        print(f"Ratio of max distance to min distance: sqrt({final_ratio_squared:.20f})")