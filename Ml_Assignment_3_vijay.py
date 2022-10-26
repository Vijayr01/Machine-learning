

# Question 1




import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt


df=pd.read_csv("/Users/vijay/Downloads/Dataset/train.csv")

df.head()


le = preprocessing.LabelEncoder()
df['Sex'] = le.fit_transform(df.Sex.values)
df['Survived'].corr(df['Sex'])


matr = df.corr()
print(matr)



df.corr().style.background_gradient(cmap="Blues")



sns.heatmap(matr, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()



#NAIVE BAYES
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore")

train = pd.read_csv('/Users/vijay/Downloads/Dataset/train.csv')
test = pd.read_csv('/Users/vijay/Downloads/Dataset/test.csv')


train['train'] = 1
test['train'] = 0
df = train.append(test, sort=False)


features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

df = df[features + [target] + ['train']]

df['Sex'] = df['Sex'].replace(["female", "male"], [0, 1])
df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
train = df.query('train == 1')
test = df.query('train == 0')


# In[56]:



# Drop missing values from the train set.
train.dropna(axis=0, inplace=True)
labels = train[target].values


train.drop(['train', target, 'Pclass'], axis=1, inplace=True)
test.drop(['train', target, 'Pclass'], axis=1, inplace=True)


# In[57]:


from sklearn.model_selection import train_test_split, cross_validate

X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)


# In[58]:


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore")


# In[59]:


classifier = GaussianNB()

classifier.fit(X_train, Y_train)


# In[60]:


y_pred = classifier.predict(X_val)

print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))

from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# Question 2

# In[42]:


glass=pd.read_csv('/Users/vijay/Downloads/Dataset/glass.csv')


# In[61]:


glass.head()


# In[62]:


glass.corr().style.background_gradient(cmap="Blues")


# In[63]:


sns.heatmap(mat, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


# In[64]:


features = ['Rl', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
target = 'Type'


X_train, X_val, Y_train, Y_val = train_test_split(glass[::-1], glass['Type'],test_size=0.2, random_state=1)

classifier = GaussianNB()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[65]:


from sklearn.svm import SVC, LinearSVC

classifier = LinearSVC()

classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_val)

# Summary of the predictions made by the classifier
print(classification_report(Y_val, y_pred))
print(confusion_matrix(Y_val, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_val, y_pred))


# In[ ]:




