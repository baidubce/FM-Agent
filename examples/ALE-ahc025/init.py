# EVOLVE-BLOCK-START
import sys

def solve():
    # 1. Read initial input: N items, D divisions, Q queries
    try:
        N, D, Q = map(int, sys.stdin.readline().split())
    except EOFError:
        return
    except ValueError:
        return

    # 2. Query Phase: Perform all Q queries
    for q in range(Q):
        # --- Strategy: Always compare item 0 and item 1 ---
        # This is a valid query, but we will ignore the result.
        # This is simpler than picking random items.
        
        print(f"1 1 0 1")
        
        # IMPORTANT: Flush stdout to send the output to the judge
        sys.stdout.flush()

        # Read the judge's response and discard it
        result = sys.stdin.readline().strip()

    # 3. Division Phase: Assign all N items to D groups
    
    # --- Strategy: Round-robin assignment ---
    # This is much simpler than sorting by score.
    # Item 0 goes to group 0 (0 % D)
    # Item 1 goes to group 1 (1 % D)
    # ...
    # Item D goes to group 0 (D % D)
    # ...

    assignments = [0] * N
    
    for i in range(N):
        assignments[i] = i % D

    # 4. Final Output
    # Print the assignments for all N items, space-separated
    print(" ".join(map(str, assignments)))
    sys.stdout.flush()

if __name__ == "__main__":
    solve()
# EVOLVE-BLOCK-END
