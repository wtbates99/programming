nums = [2,3,5,7]
def minBitwiseArray(nums): 
    ans = []

    for n in nums:
        x = 0 
        while x <= n:
            if x | (x + 1) == n:
                print(f"X: {x} bit: {x | (x + 1)} n: {n}")
                ans.append(x)
                break
            x += 1
        else:
            ans.append(-1)
    return ans

print(minBitwiseArray(nums))
