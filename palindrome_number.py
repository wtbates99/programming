x = 789
y = "racecar" 

def isPalindrome(x):
    s = str(x)
    l = len(s)

    for i in range(0,l):
        if s[i] == s[l-i-1]:
            continue
        else:
            print(f"{x} is not a palindrome")
            return 0 

    print(f"{x} is a palindrome")

isPalindrome(x)
isPalindrome(y)
