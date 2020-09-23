
import random 
random.seed(100)
b=random.random()
print(b)
""" if i specify seed value then each time run b=random.random(), i will get same value """

"""Same here each time will get different between 1 - 10"""
random.randint(1,10)








""" how to generate a random binary distribution with a random probability """

"""random library will be used"""

import random             

"""first, define a function that generates the distribution"""

def to_Get_random_distribution_with_fiexed_robability(number, probability, seed):
    
"""name the function, and giving three prositional arguements"""

"""the random.seed() here needs to be specified, because it will give the same number each time
   for functions like random.random() or random.randint()"""
   
    random.seed(seed)  """here will obtain the seed value from arguement"""
    bits = ""             """the "" sign here means a 'nothing', a void"""
    for i in range(number):
        bits += "0" if random.random() > probability else "1"  
"""this will mean a stirng "0" will add to bits(as a list), if the value generated from random.random()
is smaller than the probability given from the arguement, else will add a string "1" instead."""

"""this is a 'for' loop using range() function, and taken the value provided from the arguement, will do this operation
for 20 times"""

     return bits
"""then will return a list that contains all the string in that list"""

bs1 = toGetrandomdistributionwithrandomprobability(30, 0.3, 42)
print(bs1)








"""full pciture of code"""

import random   

def to_Get_random_distribution_with_random_robability(number, probability, seed):
    bits = ""   
    random.seed(seed)         
    for i in range(number):
        bits += "0" if random.random() > probability else "1"  
    return bits

bs1 = to_Get_random_distribution_with_random_robability(20, 0.6, 42)
print(bs1)






"""Notice"""

"""this is an exmple to show why seeded value is important"""

for bit in range(10):
    bits = random.random()
    print(bits)

"""0.21403955219994586
0.8113763075679999
0.0769033290593687
0.7326376104751333
0.8478798002406851
0.9339166565114212
0.4199537734471891
0.15249118864855093
0.38246795658857835
0.6410817719732039""" first time 

"""0.15547949981178155
0.9572130722067812
0.33659454511262676
0.09274584338014791
0.09671637683346401
0.8474943663474598
0.6037260313668911
0.8071282732743802
0.7297317866938179
0.5362280914547007""" second time 
    
"""without specifying the seed value, each time i run the above code i will get 10 set of different number 
between 0.0 - 1.0"""

random.seed(42)
for bit in range(10):
    bits = random.random()
    print(bits)
    
""" with sepcifying each time i run the code i will get the same 10 set of values"""

"""0.6394267984578837
0.025010755222666936
0.27502931836911926
0.22321073814882275
0.7364712141640124
0.6766994874229113
0.8921795677048454
0.08693883262941615
0.4219218196852704
0.029797219438070344""" first time 

"""0.6394267984578837
0.025010755222666936
0.27502931836911926
0.22321073814882275
0.7364712141640124
0.6766994874229113
0.8921795677048454
0.08693883262941615
0.4219218196852704
0.029797219438070344""" second time

