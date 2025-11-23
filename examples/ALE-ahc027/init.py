# EVOLVE-BLOCK-START
import sys

# Set a high recursion depth limit for the DFS
sys.setrecursionlimit(1000000)

# --- 1. Read All Input ---
N = int(input())
# Read horizontal walls
h = [input() for _ in range(N - 1)]
# Read vertical walls
v = [input() for _ in range(N)]
# Read dirt levels (though this minimal solution won't use them)
d = [list(map(int, input().split())) for _ in range(N)]

# --- 2. Setup for DFS ---
# Keep track of visited squares
visited = [[False for _ in range(N)] for _ in range(N)]
# Define move directions: (delta_i, delta_j)
# (0, 1) = Right, (1, 0) = Down, (0, -1) = Left, (-1, 0) = Up
DIJ = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# Corresponding move characters
DIR = "RDLU"
# To store the final path
path = []

# --- 3. DFS Function ---
def dfs(i, j):
    """
    Performs a Depth-First Search starting from square (i, j).
    Appends moves to the global 'path' list.
    """
    visited[i][j] = True  # Mark the current square as visited

    # Try all 4 directions
    for dir_index in range(4):
        di, dj = DIJ[dir_index]
        ni, nj = i + di, j + dj  # (ni, nj) is the new (i, j)

        # Check if the new square is within the grid bounds
        if 0 <= ni < N and 0 <= nj < N:
            # Check if we've already visited this square
            if not visited[ni][nj]:
                # Check for walls
                is_wall = False
                if di == 0:  # Horizontal move (Left or Right)
                    if v[i][min(j, nj)] == '1':
                        is_wall = True
                elif dj == 0:  # Vertical move (Up or Down)
                    if h[min(i, ni)][j] == '1':
                        is_wall = True
                
                # If there is no wall, proceed
                if not is_wall:
                    # 1. Record the move to the new square
                    path.append(DIR[dir_index])
                    # 2. Recursively explore from the new square
                    dfs(ni, nj)
                    # 3. Record the move to come back to the parent (i, j)
                    # (dir_index + 2) % 4 gives the opposite direction
                    path.append(DIR[(dir_index + 2) % 4])

# --- 4. Start the search and print the result ---
dfs(0, 0)
print("".join(path))
# EVOLVE-BLOCK-END