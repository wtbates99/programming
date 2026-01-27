nums = [5,2,3,1]

def minimumPairRemoval(nums):
    n = len(nums)
    min = None 
    count = 0 
    for i in range(1, n):
        d = nums[i - 1] - nums[i] 
        if min is None:
            min = d
            count += 1
        elif d < min:
            min = d 
            count += 1
    return count 

minimumPairRemoval(nums)
