arr = [4,2,1,3]

def minimumAbsDifference(arr):
    arr.sort()
    pairs = []
    min = None
    for i in range(len(arr) -1): 
        d = abs(arr[i] - arr[i + 1])
        if min is None:
            min = d
            pairs.append([arr[i], arr[i+1]])
        elif min > d:
            pairs = []
            min = d
            pairs.append([arr[i], arr[i+1]])
        elif min == d:
            pairs.append([arr[i], arr[i+1]])

    return pairs 




minimumAbsDifference(arr)
