# ئاساس Python Numpy

## 0.تەييارلىق

####   قاچىلاش ۋە قوزغۇتۇش jupyter

#### يۇپيتېرنى قاچىلاشتىن ئىلگىرى پايسون3 قاچىلانغان  بولۇشى كىرەك
##### Windows10 cmd  
Install  : pip install jupyter  
Start-up : jupyter notebook  
##### Anaaconda  
Install  : conda install jupyter notebook  
Start-up : jupyter notebook  

## 1 . قاچىلاش : pip install numpy

## 2.چاقىرىش


```python
import numpy as np
```

## 3.ئاساس Basic


```python
a = np.array([1,2,3], dtype='int32')
print(a)
## data type = int32
```

    [1 2 3]
    


```python
b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)
## data type = float64 默认
```

    [[9. 8. 7.]
     [6. 5. 4.]]
    


```python
# ئۆلچىمىنى كۆرۈش
print(a.ndim)
print(b.ndim)
```

    1
    2
    


```python
# تۈزۈلۈش شەكلى(كۆلىمىنى)نى كۆرۈش
print(a.shape)
print(b.shape)
```

    (3,)
    (2, 3)
    


```python
# ئېلمىنىت تىپىنى كۆرۈش
print(a.dtype)
print(b.dtype)
```

    int32
    float64
    


```python
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
```

    4
    8
    
    12
    48
    
    3
    3.0
    

## 4.رەت ۋە سىتونلىرىنى قايتا تەرتىپلەش ۋە تىزىش


```python
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
```

    [[ 1  2  3  4  5  6  7]
     [ 8  9 10 11 12 13 14]]
    


```python
# ئالاھېدە ئورۇندىكى ئېلنىتىنى كۆرۈش

print(a[0,0])
print(a[0,5])
print(a[1,5])
```

    1
    6
    13
    


```python
# ئالاھېدە قۇر ۋە سىتونلىرىنى بىرلا كۆرۈش

print(a[0,:])      # بىرىنچى قۇر
print(a[:,0])      # بىرىنچى سىتون
print()
print(a[:,2])      #ئىككىنچى سىتون
```

    [1 2 3 4 5 6 7]
    [1 8]
    
    [ 3 10]
    


```python
# خاس قۇر ياكى سىتونلىرىنى كۆرۈش

print(a)
print()
# [startindex:endindex:stepsize]
print(a[0,1:-1:2])
# بىرىنچى قۇردىن،ئىككىنچى ئېلمىنتتىن باشلاپ ئاخىردىن سانىغاندىكى ئىككىنچى ئېلمىنىتقىچە ھەر ئىككى قەسەم ئاتلىغاندىكى ئېلمىنتلار
```

    [[ 1  2  3  4  5  6  7]
     [ 8  9 10 11 12 13 14]]
    
    [2 4 6]
    

## ئۈچ ئۆلچەملىك ماترىسسا


```python
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)
```

    [[[1 2]
      [3 4]]
    
     [[5 6]
      [7 8]]]
    


```python
# ئېندىكىسلاپ خاس ئېلمىتىلىرىنى كۆرۈش
print(b[0,1,1])
print(b[1,1,1])
```

    4
    8
    


```python
# ئېلمىنىتلارنى ئالماشتۇرۇش
print(b)
print("-----------------------")
b[:,1,:] = [[9,9],[8,8]]
print(b)
```

    [[[1 2]
      [3 4]]
    
     [[5 6]
      [7 8]]]
    -----------------------
    [[[1 2]
      [9 9]]
    
     [[5 6]
      [8 8]]]
    

## ئالاھېدە ماترىسسالار


```python
# نۆل ماترىسسا
np.zeros([3,4],dtype='int32')     # np.zeros((3,4),dtype='int32')
```




    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])




```python
# بىر ماترىسسا
print(np.ones([3,4],dtype='int32'))     # np.ones((3,4),dtype='int32')
print("------------------------")
print(np.ones([2,3,4],dtype='int32'))     # np.ones((2,3,4),dtype='int32')
```

    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]
    ------------------------
    [[[1 1 1 1]
      [1 1 1 1]
      [1 1 1 1]]
    
     [[1 1 1 1]
      [1 1 1 1]
      [1 1 1 1]]]
    


```python
# ئالاھىدە سان تولدۇرۇلغان ماترىسسا
# np.full(size,number)

print(np.full([3,4],99))    # np.full((3,4),99)
```

    [[99 99 99 99]
     [99 99 99 99]
     [99 99 99 99]]
    


```python
# باشقا ماترىسسىغا ئوخشاش كۆلەمدە سان تولدۇرۇش
print(a)
print("---------------------------")
np.full_like(a, 4)
```

    [[ 1  2  3  4  5  6  7]
     [ 8  9 10 11 12 13 14]]
    ---------------------------
    




    array([[4, 4, 4, 4, 4, 4, 4],
           [4, 4, 4, 4, 4, 4, 4]])




```python
# خالىغان قىممەتتىكى ماترىسىنى ھاسىل قىلىش

# np.random.rand(size)  ئېلمىنىتلىرى 0~1 بولغان خالىغان قىممەت

print(np.random.rand(4,2))
```

    [[0.2353494  0.4123137 ]
     [0.17289768 0.44002922]
     [0.81088531 0.01156963]
     [0.45841134 0.13829674]]
    


```python
# np.random.randint(a,b, size) 
# بولغان خالىغان پۈتۈن سانلىق قىممەتتىكى ماترىسسا a~b ئېلمىنىتلىرى

print(np.random.randint(2,6, size=(3,3)))
```

    [[3 2 3]
     [5 3 4]
     [5 5 2]]
    


```python
# بىرلىك ماترىسسا(بىر ماترىسسا ئەمەس)
# np.identity(size,dtype='')

print(np.identity(5,dtype='int32'))
```

    [[1 0 0 0 0]
     [0 1 0 0 0]
     [0 0 1 0 0]
     [0 0 0 1 0]
     [0 0 0 0 1]]
    


```python
# تەكرارلاش
ar = np.array([[1,2,3]])
print(ar)
print("------------------------")
# np.repeat(array, repeat-times, axis=0)
br = np.repeat(ar,5, axis=0)
print(br)
```

    [[1 2 3]]
    ------------------------
    [[1 2 3]
     [1 2 3]
     [1 2 3]
     [1 2 3]
     [1 2 3]]
    


```python
# نۇسخىلاش

a = np.array([1,2,3])
b = a.copy()
b[0] = 100
print(a)
print(b)
```

    [1 2 3]
    [100   2   3]
    

## ماتىماتىكىلىق ھېساپلاشلار


```python
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
```

    [1 2 3 4]
    [1 0 1 0]
    ---------------------
    [3 4 5 6]
    [-1  0  1  2]
    [2 4 6 8]
    [0.5 1.  1.5 2. ]
    [ 1  4  9 16]
    ------------------------
    [2 2 4 4]
    [0 2 2 4]
    [1 0 3 0]
    


```python
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
```

    [1 2 3 4]
    ---------------------
    [ 0.54030231 -0.41614684 -0.9899925  -0.65364362]
    [ 0.84147098  0.90929743  0.14112001 -0.7568025 ]
    [ 1.55740772 -2.18503986 -0.14254654  1.15782128]
    ------------------------
    [[1. 1. 1.]
     [1. 1. 1.]]
    [[2 2]
     [2 2]
     [2 2]]
    سىزىقلىق ئالگېبرالىق كۆپەيتىشلەر
    [[6. 6.]
     [6. 6.]]
    
    [[6. 6.]
     [6. 6.]]
    


```python
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
```

    [[6 9 8]
     [8 9 4]
     [1 7 3]]
    
    190.0
    

## سىتاستىكىلىق ھېساپلاش


```python
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
```

    [[1 2 3]
     [4 5 6]]
    ----------------------
    1
    [1 2 3]
    [1 4]
    
    6
    
    21
    [5 7 9]
    [ 6 15]
    


```python
# قېلىپلاشتۇرۇش
#np.reshape(size)

c = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(c)
print()
print(c.reshape((4,3)))
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    
    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    


```python
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
```

    [[1 2 3 4]
     [5 6 7 8]
     [1 2 3 4]
     [5 6 7 8]]
    
    [[1. 1. 1. 1. 0. 0.]
     [1. 1. 1. 1. 0. 0.]]
    


```python
# ھۆجقەتتىن ئوقۇش

#filedata = np.genfromtxt('data.txt', delimiter=',')
#filedata = filedata.astype('int32')
#print(filedata)
# (~((filedata > 50) & (filedata < 100)))
```


```python

```


```python

```
