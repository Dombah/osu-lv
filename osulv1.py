''' 1.zad
print("Radni sati: ")
hours = int(input().split('h')[0])
print("eur/per hour")
hourly_salary = float(input())

print(f"Ukupno: {hours*hourly_salary} eura")
'''


''' 2.zad
try:
    grade = float(input())
    while grade < 0.0 or grade > 1.0:
        grade = float(input())
    if grade >= 0.9:
        print("A")
    elif grade >= 0.8:
            print("B")
    elif grade >= 0.7:
        print("C")
    elif grade >= 0.6:
        print("D")
    else:
        print("F")
except:
    print("Non number value entered")
'''

''' 3.zad
numbers = []
while True:
    inp = input()
    if inp == "Done":
        break
    else:
        try:
            inp = float(inp)
            numbers.append(inp)
        except:
            continue

print(f"User entered: {len(numbers)} numbers")
numbers.sort()
print(f"Maximum number: {numbers[-1]}")
print(f"Minimum number: {numbers[0]}")

sum = 0
for number in numbers:
    sum += number
print(f"Average of numbers: {sum / len(numbers)}")
'''

''' 4. zad
songs = open('song.txt')
words = []
dict = {}
for line in songs:
    line = line.rstrip()
    words_in_line = line.split(' ')
    words_in_line = words_in_line
    words += words_in_line

unique_words = list(set(words))

for unique_word in unique_words:
    count = 0
    for word in words:
        if word == unique_word:
            count += 1
    dict[unique_word] = count

print(dict)

count = 0
one_appearance_words = []
for k,v in dict.items():
    if(v == 1):
        one_appearance_words.append(k)
        count += 1

print(f"Count: {count}")
print("Words appearing one time")
print(one_appearance_words)
'''

messages = open('SMSSpamCollection.txt')

dict = {}
count_ham = 0
count_spam = 0

words_ham = []
words_spam = []

ending_with_count = 0

for line in messages:
    line = line.rstrip()
    if line[0:3] == 'ham':
        count_ham += 1
        temp_words = line.split(' ')
        temp_words.pop(0)
        words_ham += temp_words
        continue
    elif line[0:4] == 'spam':
        count_spam += 1
        temp_words = line.split(' ')
        temp_words.pop(0)
        words_spam += temp_words
        if line.endswith('?'): 
            ending_with_count += 1

print("Average of ham words: ", round(len(words_ham) / count_ham, 2) )
print("Average of spam words: ", round(len(words_spam) / count_spam, 2))

print("Spam sentences ending with ?: ", ending_with_count)
    
