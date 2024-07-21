""" String """
var = 'JV'
print(var*3)  #Repetition
print(type(var))

var4 = "Olivu, Jayavaradhan"
print(var4.split(','))
var5 = '     Olivu, Jayavaradhan, Varadhan, Jaya'
print(var5.strip())
''' formating the strings '''

Name = 'JV'
Age = 22
Line1 = 'My Name is {} and I am {} age old.'.format(Name, Age)
print(Line1)
Line2 = f'My Name is {Name} and I am {Age} age old.'
print(Line2)
print('---------------------')

var1 = [12, 'Dhoni', ['a', 'b'], 55.66]
print(var1)
print(type(var1))

# var2 = 'Dhoni'
# var2[4] = 'y'
# print(var2[3])
# print(var2)

var3 = ['abc', 'cfc', 'jv', 'dd']
var3[0] = 'JV1'
print(var3)
print(var3[3][1])  #it prints 1st index which is 'd' from the third index data 'dd'

# var4 = 'JV'
# AA = dir(var4)
# print(AA)

Tuple1 = ('JV', 651, 'Sreeram')
print(type(Tuple1))

dictionary = {"Name": "Jayavaradhan", "Team": "Business Operation", "Age": 21}
print(dictionary)
print(dictionary['Age'])
print(dictionary.keys())
print(dictionary.values())

dictionary_1 = {"Team Members": ["Jayavaradhan", "Sreeram"], "Team": "Business Operation", "Age": [21, 30]}
print(dictionary_1)
print(dictionary_1['Age'])
print(dictionary_1['Team Members'])
print(dictionary_1.keys())
print(dictionary_1.values())

dictionary_1['Team Members'][0] = 'JV'
print(dictionary_1)

''' Numbers '''

x = 35e3
y = 45332543
z = -5j
print(type(x))  # Float
print(type(y))  #int
print(type(z))  #complex


''' Random '''

import random

print(random.randrange(1, 50))

''' list '''

list1 = ['JV', 'Jaya', 'Varadhan']
list1.append('Jayavaradhan')  # will at last
list1.extend(['a', 'b'])  # used to add more than one data
list1.insert(0, 'Olivu')  # based on the index we can add the data

print(list1)

# list1.remove('JV')  # removing a specific
# print(list1)
# list1.clear()
# print(list1)

# del list1
# print(list1)  # will receive an error since we remove the list1

''' storing in list '''

list1.sort(reverse=True)
print(list1)
list1.sort()
print(list1)

''' for truple and dictionary please refer the Dashboard notes '''




