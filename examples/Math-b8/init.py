# EVOLVE-BLOCK-START
"""
An initial program
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def calculate_ratio(points):
    """Calculate the ratio between maximum and minimum pairwise distances."""
    if len(points) < 2:
        return float('inf')
    
    # Calculate all pairwise distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return float('inf')
    
    d_max = np.max(distances)
    d_min = np.min(distances)
    
    if d_min == 0:
        return float('inf')
    
    return d_max / d_min

def construct_16_points():
    """Construct 16 points in 2D space to minimize the ratio d_max/d_min."""
    # Start with a hexagonal-like pattern as initial configuration
    points = []
    center = [2.5, 2.5]
    
    # Inner ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        x = center[0] + 1.2 * np.cos(angle)
        y = center[1] + 1.2 * np.sin(angle)
        points.append([x, y])
    
    # Outer ring (10 points)
    for i in range(10):
        angle = i * 2 * np.pi / 10
        x = center[0] + 2.0 * np.cos(angle)
        y = center[1] + 2.0 * np.sin(angle)
        points.append([x, y])
    
    points = np.array(points)
    
    # Optimize the entire configuration to minimize the ratio
    def objective_function(point_params):
        points_reshaped = point_params.reshape(-1, 2)
        return calculate_ratio(points_reshaped)
    
    # Optimize all points simultaneously
    initial_params = points.flatten()
    
    result = minimize(
        objective_function,
        initial_params,
        method='L-BFGS-B',
        options={'maxiter': 300}
    )
    
    if result.success:
        points = result.x.reshape(-1, 2)
    
    return points

def run_construction():
    """Main function that runs the construction and returns results."""
    try:
        points = construct_16_points()
        return points
    except Exception as e:
        return None
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Run the construction
    points = run_construction()
    ratio = calculate_ratio(points)
    
    if points is not None:
        print(f"Ratio: √{ratio:.20f}")
        print(f"Target: √12.889266112 ≈ {np.sqrt(12.889266112):.20f}")
        
    else:
        print("Construction failed.")
    #@title Construction 1: verification
    import scipy as sp

    print(f'Construction 1 has {len(points)} points in {points.shape[1]} dimensions.')
    pairwise_distances = sp.spatial.distance.pdist(points)
    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)

    ratio_squared = (max_distance / min_distance)**2
    print(f"Ratio of max distance to min distance: sqrt({ratio_squared})")
