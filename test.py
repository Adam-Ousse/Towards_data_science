from numpy import*
def saisie():
    x = int(input('donner un entier'))
    while not 2<=x<=20 :
        x = int(input('donner un entier'))
    return(x)
def remplir(v,n):
    v[0] = int(input('donner la premier entier'))
    for i in range(1,n):
        v[i] = int(input("donner entier numero " + str(i)))
        while not v[i-1]<v[i]:
                v[i] = int(input("donner entier numero " + str(i)))
def cases(v2,v1,m,n):
     i =0
     j=0
     x=0
     while (i!=n and j!=m) :
          if v1[i]<v2[j]:
               i = i+1
               x = x+1
          elif v2[j]<v1[i]:
               j = j+1
               x = x+1
          else:
               j = j+1
               x = x+1
               i = i+1
     if i == n:
          for t in range (j,m):
               x = x+1
     else:
          for t in range (i,n):
               x = x+1
     return(x)
def affiche(v3,x):
     for i in range(x):
          print(v3[i])

def fusion(v3,v2,v1,m,n):
     i =0
     j=0
     x=0
     while i!=n and j!=m :
          if v1[i]<v2[j]:
               v3[x] = v1[i]
               i = i+1
               x = x+1
          elif v2[j]<v1[i]:
               v3[x] = v2[j]
               j = j+1
               x = x+1
          else:
               v3[x] = v2[j]
               j = j+1
               x = x+1
               i = i+1
     if i == n:
          for t in range (j,m):
               v3[x] = v2[j]
               x = x+1
     else:
          for t in range (i,n):
               v3[x] = v1[i]
               x = x+1
def affiche(v3,x):
     for i in range(x):
          print(v3[i])









n = saisie()
m = saisie()
V1 = array([int()]* n)
V2 = array([int()]*m)
print(V1.shape)
print(V2.shape)
remplir(V1,n)
remplir(V2,m)
p = cases(V2,V1,m,n)
V3 = array([int()]*p)
fusion(V3,V2,V1,m,n)
affiche(V3,p)