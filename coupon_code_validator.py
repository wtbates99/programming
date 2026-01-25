import re

def validateCoupons(code, businessLine, isActive):
    valid = []
    for i in range(len(code)):  # i is now an integer index
        if code[i] is not None and re.fullmatch(r'\w+', code[i]) and isActive[i] == True:
            valid.append(code[i])
        else:
            print(f"invalid {code[i]}")

    return valid

code = ["SAVE20","","PHARMA5","SAVE@20"]
businessLine = ["restaurant","grocery","pharmacy","restaurant"]
isActive = [True,True,True,True]

print(validateCoupons(code, businessLine, isActive))

