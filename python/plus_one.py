digits = [9,9]

def plusOne(digits):
    number = int(''.join(str(digit) for digit in digits))
    number += 1
    l = []
    for i in str(number):
        l.append(int(i))

    return l 

print(plusOne(digits))
