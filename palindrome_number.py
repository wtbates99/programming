x = 789
y = 12122

def isPalindrome(x):
    s = str(x)
    l = len(s)

    for i in range(0,l):
        if s[i] == s[l-i-1]:
            continue
        else:
            print('Not a isPalindrome')
            return 0 

    print("Is a palindrome")

isPalindrome(x)
isPalindrome(y)
