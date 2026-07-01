from collections import defaultdict


def count_safe_intervals(n, m, vulnerable_bacteria, toxic_bacteria):
    """
    Count all contiguous intervals [l, r] of bacteria samples where no toxic
    relationship exists between any pair within that interval.

    Toxic relationships are one-way in cause, but co-existence is mutually
    forbidden: if A is toxic to B, neither can share an interval with the other.

    Args:
        n: total number of bacteria samples, labeled 1..n placed in that order
        m: number of toxic relationships
        vulnerable_bacteria: list of length m — vulnerable_bacteria[i] is harmed by toxic_bacteria[i]
        toxic_bacteria:      list of length m — toxic_bacteria[i] harms vulnerable_bacteria[i]

    Returns:
        int: number of valid (conflict-free) contiguous intervals
    """
    # Treat each toxic relationship as an undirected co-existence conflict.
    conflicts = defaultdict(list)
    for toxic, vulnerable in zip(toxic_bacteria, vulnerable_bacteria):
        conflicts[toxic].append(vulnerable)
        conflicts[vulnerable].append(toxic)

    left = 1
    count = 0

    for right in range(1, n + 1):
        # Push `left` past any bacteria in the current window that conflicts
        # with `right`.
        for b in conflicts[right]:
            if left <= b < right:
                left = b + 1

        # Every interval [left, right], [left+1, right], ..., [right, right]
        # ends at `right` and is conflict-free.
        count += right - left + 1

    return count


# ---------------------------------------------------------------------------
# Example / quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # n=7 bacteria in a row: [1, 2, 3, 4, 5, 6, 7]
    # toxic relationships: 1→3, 2→5, 4→6
    n = 7
    toxic_bacteria      = [1, 2, 4]
    vulnerable_bacteria = [3, 5, 6]
    m = len(toxic_bacteria)

    result = count_safe_intervals(n, m, vulnerable_bacteria, toxic_bacteria)
    print(f"Number of safe intervals: {result}")

    # Brute-force verification
    conflict_set = set(zip(toxic_bacteria, vulnerable_bacteria))
    conflict_set |= set(zip(vulnerable_bacteria, toxic_bacteria))
    brute = sum(
        1
        for l in range(1, n + 1)
        for r in range(l, n + 1)
        if all((a, b) not in conflict_set for a in range(l, r + 1) for b in range(l, r + 1))
    )
    print(f"Brute-force check:        {brute}")
    assert result == brute, "Mismatch!"
    print("OK")
