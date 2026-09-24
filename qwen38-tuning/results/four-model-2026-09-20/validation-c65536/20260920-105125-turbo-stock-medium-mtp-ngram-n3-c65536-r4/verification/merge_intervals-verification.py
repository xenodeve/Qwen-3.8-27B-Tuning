def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for interval in intervals[1:]:
        last = merged[-1]
        if interval[0] <= last[1]:
            last[1] = max(last[1], interval[1])
        else:
            merged.append(interval)
    
    return merged


# ---- verification ----

assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([]) == []
assert merge_intervals([[1,4],[0,4]]) == [[0,4]]
assert merge_intervals([[1,4],[2,3]]) == [[1,4]]
assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]
