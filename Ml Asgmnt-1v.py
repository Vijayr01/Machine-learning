#!/usr/bin/env python
# coding: utf-8

# QUESTION-1

# In[12]:


ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]
# sorted() is an inbuilt method that is used to arrange the data in order.
l=sorted(ages)
print(l)
print("minimum age in the list= ", min(ages))
print("maximum age in the list= ", max(ages))

# max() and min() are the methods that gives us highest and lowest value in the given data
l.insert(1,min(ages))
l.insert(12,max(ages))
# insert() adds the data at the given index in the list.
print(l)


# In[17]:


import statistics
statistics.median(ages)


# In[18]:


statistics.mean(ages)
# mean() and median() are inbuilt methods that gives the result directly without mathametical calculations.


# In[20]:


range= max(ages)- min(ages)
print("Range is:", range)


# QUESTION-2

# In[13]:


dog = {}
#dictionary  dog is created.
dog = {'name': 'Rampo', 'colour': 'black', 'breed':'Lab', 'legs': '4', 'age':'3'}
print(dog)


# In[35]:


student_dict = {}
student_dict = { 'first_name', 'last_name', 'gender', 'age', 'marital status', 'Skills', 'Country', 'City', 'Address'}
print(student_dict)
print(len(student_dict))
# len() method returns the length of the data.


# In[14]:


student_dict = { 'first_name':'Vijay', 'last_name': 'Varkuti', 'gender':'M', 'age':'23','marital status':'Single', 'skills':["C","C++","Python"],'Country':'United States', 'City':'Willow Creek','Address':'Wornall Road'}
print(student_dict)
print(student_dict.get('skills'))
# .get() method returns the value of the specified key.
print(type(student_dict.get('skills')))
student_dict['skills'].append('XML')
print(student_dict.get('skills'))


# In[41]:


d_keys= student_dict.keys()
# keys() returns all the keys in the dictionary.
print(keys)


# In[16]:


d_values= student_dict.values()
# values() returns all the values in the dictionary.
print(d_values)


# QUESTION-3

# In[17]:


sisters = (" Sravya", "Kavya", "Bindu", "Sindhu")
brothers = ("Karthik", "Bahu", "Aryan", "Johny")
Siblings= sisters + brothers
# '+' joins both the tuples.
print(Siblings)


# In[21]:


print(len(Siblings))
family_members = list(Siblings)
# The above list is assigned to variable name family_members
family_members.append("Srinivas")
family_members.append("Jagadeeshwari")
print(family_members)


# QUESTION-4

# In[25]:


it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'}     
print(len(it_companies))


# In[75]:


it_companies.add("Twitter")  
# add() is used to add data to the present dataset.
print(it_companies)


# In[28]:


Multiple_It_Companies = {'Infosys', 'Legato', 'TCS'}    
it_companies.update(Multiple_It_Companies)     
# update() is used to add data to a set from another set
print(it_companies)


# In[76]:


it_companies.discard('TCS')
# discard() is used to remove a dataitem from the set and and also it doesn't raise an error if the parameter is missing.
print(it_companies)
# The difference between remove() and discard() is that remove() will raise an error if the specified value is not present and the discrad() will not raise an error()


# In[30]:


A = {19,22,24,20,25,26}
      
B= {19,22,20,25,26,24,28,27}  
Union = A.union(B)
# union() returns the combined set of both the sets without repetiting elements.     
print(Union)


# In[31]:


Intersection = A.intersection(B)
# intersection() returns the set of elements that are in common in both the sets.      
print(Intersection)


# In[33]:


Subset = A.issubset(B)
# issubset() returns whether A is a subset of B
print(Subset)


# In[34]:


Disjoint = A.isdisjoint(B)  
# isdisjoint() returns true if there is no intersection in the data or else returns False.
print(Disjoint)


# In[69]:


A = {19,22,24,20,25,26}
B= {19,22,20,25,26,24,28,27} 
AUB= A.union(B)      
BUA= B.union(A)      
print(AUB)
print(BUA)


# In[70]:


Sym_diff = A.symmetric_difference(B)     
# symmetric_difference() returns a set with the symmetric difference of both the sets.
print(Sym_diff)


# In[71]:


print(A.clear())
print(B.clear())


# In[42]:


age = [22,19,24,25,26,24,25,24]
print(len(age))


# In[17]:


age_set = set(age)     
print(age_set)
print(len(age_set))


# QUESTION-5

# In[18]:


radius =  30      
_area_of_circle_ = 3.14*30*30  
print(_area_of_circle_)


# In[49]:


radius = 30      
_circum_of_circle_ = 2*3.14*30   
print(_circum_of_circle_)


# In[51]:


radius = int(input("Enter radius of the circle: "))
# takes the value from user
area_of_circle = 3.14*radius*radius      
print("Area is: ", area_of_circle)


# QUESTION 6

# In[62]:


sentence = "I am a teacher and I love to inspire and teach people"
a =(sentence.split())
print(a)
# split() method divides the words individually in the sentence.
b=set(a)
print(b)
print(len(b))


# QUESTION 7

# In[73]:


print("Name\t\tAge\tCountry\t  City")
# \t in python is used for a tab space.
print("Asabeneh\t250\tFinland\t  Helsinki")  


# In[27]:


radius = 10
area = 3.14 * radius ** 2
input = "The area of a circle with radius {} is {} meters square."
print(inp.format(radius,area))
# The format() method inserts the user specified value into the string's placeholder.
      


# QUESTION 9

# In[65]:


N = int(input("Enter the number of students: "))
weights_lbs = []
print("Enter the weights of {} students".format(N))
for i in range(0, N):
    weight_lb = int(input())
    weights_lbs.append(weight_lb)
print("{} students weights in lbs {}".format(N, weights_lbs))
weights_kgs = [i * 0.453592 for i in weights_lbs]
print(N, "students weights in kgs", weights_kgs)


# In[ ]:





# In[ ]:




