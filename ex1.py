import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', action="store", dest="task", default=False)
parser.add_argument('--arg', action="store", dest="arg", default='1')
results = parser.parse_args()


# this function count how many vowels in a string
def vowels_count(vowel):
    n = 0
    for j in range(0, len(vowel)):
        if vowel[j] in 'AaEeIiUuYyOo':
            n += 1
    return n


# this function make a^b
def power_multiplication(a, b):
    result = 1
    while b > 0:
        if b % 2 == 1:
            result = result * a
            b = b - 1
        else:
            a = a * a
            b = b // 2
    return result


# this function create array with the perfect numbers with index between 1 to 100
def Perfect_power(n):
    min_base = 2
    max_base = 80
    min_exp = 2
    max_exp = 11
    power_list = [0, 1]
    for i in range(min_base, max_base + 1):
        for j in range(min_exp, max_exp + 1):
            if power_multiplication(i, j) <= 7000:
                power_list.append(power_multiplication(i, j))
    power_list.sort()
    power_list = list(dict.fromkeys(power_list))
    return power_list[n]


def Lazy_caterer(n):
    p = ((n - 1) ** 2 + (n - 1) + 2) // 2
    return p


if results.task == 'vowels':
    print(vowels_count(results.arg))
elif results.task == 'perfect':
    print(Perfect_power(int(results.arg)))
elif results.task == 'lazy':
    print(Lazy_caterer(int(results.arg)))
else:
    print('wrong input')
