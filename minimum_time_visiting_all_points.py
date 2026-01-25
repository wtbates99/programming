points = [[1,1], [3,4], [-1,0]]
def minTimeToVisitAllPoints(points):
    dist = 0
    for i in range(len(points)-1):
        x = (points[i][0])
        fx = points[i+1][0]
        dx = abs(x - fx)
        y = (points[i][1])
        fy = points[i+1][1]
        dy = abs(y - fy)
        dist += max(dx, dy)

    return dist

print(minTimeToVisitAllPoints(points))
