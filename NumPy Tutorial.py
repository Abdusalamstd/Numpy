#!/usr/bin/env python
# coding: utf-8

# # ئاساس Python Numpy

# ## 0.تەييارلىق

# ####   قاچىلاش ۋە قوزغۇتۇش jupyter

# #### يۇپيتېرنى قاچىلاشتىن ئىلگىرى پايسون3 قاچىلانغان  بولۇشى كىرەك
# ##### Windows10 cmd  
# Install  : pip install jupyter  
# Start-up : jupyter notebook  
# ##### Anaaconda  
# Install  : conda install jupyter notebook  
# Start-up : jupyter notebook  

# ## 1 . قاچىلاش : pip install numpy

# ## 2.چاقىرىش

# In[127]:


import numpy as np


# ## 3.ئاساس Basic

# In[128]:


a = np.array([1,2,3], dtype='int32')
print(a)
## data type = int32


# In[129]:


b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)
## data type = float64 默认


# In[130]:


# ئۆلچىمىنى كۆرۈش
print(a.ndim)
print(b.ndim)


# In[131]:


# تۈزۈلۈش شەكلى(كۆلىمىنى)نى كۆرۈش
print(a.shape)
print(b.shape)


# In[132]:


# ئېلمىنىت تىپىنى كۆرۈش
print(a.dtype)
print(b.dtype)


# In[133]:


# چوڭلۇقىنى كۆرۈش
print(a.itemsize)
print(b.itemsize)

print()

# بارلىق چوڭلۇقىنى كۆرۈش
print(a.nbytes)
print(b.nbytes)

# ئېلمىېنىتلار سانى
print()
print(a.size)
print(a.nbytes/a.itemsize)


# ## 4.رەت ۋە سىتونلىرىنى قايتا تەرتىپلەش ۋە تىزىش

# In[134]:


a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)


# In[135]:


# ئالاھېدە ئورۇندىكى ئېلنىتىنى كۆرۈش

print(a[0,0])
print(a[0,5])
print(a[1,5])


# In[136]:


# ئالاھېدە قۇر ۋە سىتونلىرىنى بىرلا كۆرۈش

print(a[0,:])      # بىرىنچى قۇر
print(a[:,0])      # بىرىنچى سىتون
print()
print(a[:,2])      #ئىككىنچى سىتون


# In[137]:


# خاس قۇر ياكى سىتونلىرىنى كۆرۈش

print(a)
print()
# [startindex:endindex:stepsize]
print(a[0,1:-1:2])
# بىرىنچى قۇردىن،ئىككىنچى ئېلمىنتتىن باشلاپ ئاخىردىن سانىغاندىكى ئىككىنچى ئېلمىنىتقىچە ھەر ئىككى قەسەم ئاتلىغاندىكى ئېلمىنتلار


# ## ئۈچ ئۆلچەملىك ماترىسسا

# In[138]:


b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)


# In[139]:


# ئېندىكىسلاپ خاس ئېلمىتىلىرىنى كۆرۈش
print(b[0,1,1])
print(b[1,1,1])


# In[140]:


# ئېلمىنىتلارنى ئالماشتۇرۇش
print(b)
print("-----------------------")
b[:,1,:] = [[9,9],[8,8]]
print(b)


# ## ئالاھېدە ماترىسسالار

# In[141]:


# نۆل ماترىسسا
np.zeros([3,4],dtype='int32')     # np.zeros((3,4),dtype='int32')


# In[142]:


# بىر ماترىسسا
print(np.ones([3,4],dtype='int32'))     # np.ones((3,4),dtype='int32')
print("------------------------")
print(np.ones([2,3,4],dtype='int32'))     # np.ones((2,3,4),dtype='int32')


# In[143]:


# ئالاھىدە سان تولدۇرۇلغان ماترىسسا
# np.full(size,number)

print(np.full([3,4],99))    # np.full((3,4),99)


# In[144]:


# باشقا ماترىسسىغا ئوخشاش كۆلەمدە سان تولدۇرۇش
print(a)
print("---------------------------")
np.full_like(a, 4)


# In[145]:


# خالىغان قىممەتتىكى ماترىسىنى ھاسىل قىلىش

# np.random.rand(size)  ئېلمىنىتلىرى 0~1 بولغان خالىغان قىممەت

print(np.random.rand(4,2))


# In[146]:


# np.random.randint(a,b, size) 
# بولغان خالىغان پۈتۈن سانلىق قىممەتتىكى ماترىسسا a~b ئېلمىنىتلىرى

print(np.random.randint(2,6, size=(3,3)))


# In[147]:


# بىرلىك ماترىسسا(بىر ماترىسسا ئەمەس)
# np.identity(size,dtype='')

print(np.identity(5,dtype='int32'))


# In[148]:


# تەكرارلاش
ar = np.array([[1,2,3]])
print(ar)
print("------------------------")
# np.repeat(array, repeat-times, axis=0)
br = np.repeat(ar,5, axis=0)
print(br)


# In[149]:


# نۇسخىلاش

a = np.array([1,2,3])
b = a.copy()
b[0] = 100
print(a)
print(b)


# ## ماتىماتىكىلىق ھېساپلاشلار

# In[150]:


# ئاساس
a = np.array([1,2,3,4])
b = np.array([1,0,1,0])

print(a)
print(b)
print("---------------------")
print(a+2)
print(a-2)
print(a*2)
print(a/2)
print(a**2)
print("------------------------")
print(a+b)
print(a-b)
print(a*b)


# In[151]:


# تىرگونۇمىترىيەلىك فۇنكىسيە ۋە باشقا فۇنكىسيەلەر

print(a)
print("---------------------")
print(np.cos(a))
print(np.sin(a))
print(np.tan(a))
# .....

print("------------------------")
a = np.ones((2,3))
print(a)
b = np.full((3,2), 2)
print(b)
print("سىزىقلىق ئالگېبرالىق كۆپەيتىشلەر")
print(a.dot(b))
print()
print(np.matmul(a,b))


# In[152]:


# كۇۋادىرات ماترىسسانىڭ دېترمىنانتىنى تېپىش

c = np.random.randint(1,10,size=(3,3))
print(c)
print()
print(np.linalg.det(c))

# Determinant
# Trace
# Singular Vector Decomposition
# Eigenvalues
# Matrix Norm
# Inverse
# Etc...


# ## سىتاستىكىلىق ھېساپلاش

# In[153]:


st = np.array([[1,2,3],[4,5,6]])
print(st)
print("----------------------")
print(np.min(st)) # ئەڭ كىچىك ئېلمىنىت
print(np.min(st,axis=0)) # ھەر قايسى سىتوندىكى ئەڭ كىچىك ئېلمىنىتلار
print(np.min(st,axis=1)) # ئىككىنچى قۇردىكى ئەڭ كىچىك ئېلمىنتلار
print()
print(np.max(st)) # ئەڭ چوڭ ئېلمىنت
print()
print(np.sum(st)) #ئېلمىنىتلار يىغىندىسى
print(np.sum(st,axis=0)) # ھەرقايسى سىتوندىكى ئېلمىنىتلار يىغىندىسى
print(np.sum(st,axis=1)) #...........


# In[154]:


# قېلىپلاشتۇرۇش
#np.reshape(size)

c = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(c)
print()
print(c.reshape((4,3)))


# In[157]:


# ۋېرتىكال ۋە گوروزونتاللانغان ماترىسسا

# Vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
print(np.vstack([v1,v2,v1,v2]))

print()

# Horizontal  stack
h1 = np.ones((2,4))
h2 = np.zeros((2,2))
print(np.hstack((h1,h2)))


# In[156]:


# ھۆجقەتتىن ئوقۇش

#filedata = np.genfromtxt('data.txt', delimiter=',')
#filedata = filedata.astype('int32')
#print(filedata)
# (~((filedata > 50) & (filedata < 100)))


# In[ ]:





# In[ ]:




