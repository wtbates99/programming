nums = [9, 4, 1, 7, 100, 72, 99, 68, 29]
k = 2
def minimumDifference(nums, k):
    if k <= 1:   # edge case
        return 0

    nums.sort()  # the key step

    d = None     # to track the minimum difference

    # loop over all contiguous slices of length k
    for i in range(len(nums) - k + 1):
        new_diff = nums[i + k - 1] - nums[i]  # max - min in this window
        print("Index: ", i, "\n New Diff: ", new_diff, "\n Max: ", nums[i+k-1], "\n Min: ", nums[i])
        if d is None or new_diff < d:
            d = new_diff

    return d

print(minimumDifference(nums, k))

