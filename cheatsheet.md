#### 汉诺塔
```python
def move(k,i1,i2,i3,num):
    if k == 1:
        print(f'{num[0]}:{i1}->{i2}')
        return
    else:
        move(k-1,i1,i3,i2,num[:-1])
        move(1,i1,i2,i3,[num[-1]])
        move(k-1,i3,i2,i1,num[:-1])

n,i1,i2,i3=input().split()
n=int(n)
move(n,i1,i3,i2,[i for i in range(1,n+1)])
```

#### 逆波兰表达式求值

```python
stack=[]
for t in s:
    if t in '+-*/':
        b,a=stack.pop(),stack.pop()
        stack.append(str(eval(a+t+b)))
    else:
        stack.append(t)
print(f'{float(stack[0]):.6f}')
```

#### 中序表达式转后序表达式


```python
#前缀转后缀
def infix_to_postfix(expression):
    def get_precedence(op):
        precedences = {'+': 1, '-': 1, '*': 2, '/': 2}
        return precedences[op] if op in precedences else 0

    def is_operator(c):
        return c in "+-*/"

    def is_number(c):
        return c.isdigit() or c == '.'

    output = []
    stack = []
    number_buffer = []
    
    def flush_number_buffer():
        if number_buffer:
            output.append(''.join(number_buffer))
            number_buffer.clear()
	
    # 主体部分
    for c in expression:
        if is_number(c):
            number_buffer.append(c)
        elif c == '(':
            flush_number_buffer()
            stack.append(c)
        elif c == ')':
            flush_number_buffer()
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # popping '('
        elif is_operator(c):
            flush_number_buffer()
            while stack and get_precedence(c) <= get_precedence(stack[-1]):
                output.append(stack.pop())
            stack.append(c)

    flush_number_buffer()
    while stack:
        output.append(stack.pop())

    return ' '.join(output)

# Read number of expressions
n = int(input())
# Read each expression and convert it
for _ in range(n):
    infix_expr = input()
    postfix_expr = infix_to_postfix(infix_expr)
    print(postfix_expr)
```
中缀转后缀
```python
operators=['+','-','*','/']
cals=['(',')']
# 预处理数据的部分已省略。
def pre_to_post(lst):
    s_op,s_out=[],[]
    while lst:
        tmp=lst.pop(0)
        if tmp not in operators and tmp not in cals:
            s_out.append(tmp)
            continue

        if tmp=="(":
            s_op.append(tmp)
            continue

        if tmp==")":
            while (a:=s_op.pop())!="(":
                s_out.append(a)

        if tmp in operators:
            if not s_op:
                s_op.append(tmp)
                continue
            if is_prior(tmp,s_op[-1]) or s_op[-1]=="(":
                s_op.append(tmp)
                continue
            while (not (is_prior(tmp,s_op[-1]) or s_op[-1]=="(")
                or not s_op):
                s_out.append(s_op.pop())
            s_op.append(tmp)
            continue

    while len(s_op)!=0:
        tmp=s_op.pop()
        if tmp in operators:
            s_out.append(tmp)

    return " ".join(s_out)

def is_prior(A,B):
    if (A=="*" or A=="/") and (B=="+" or B=="-"):
        return True
    return False

def input_to_lst(x):
    tmp=list(x)

for i in range(int(input())):
    print(pre_to_post(expProcessor(input())))
```

#### 最大全0子矩阵

```python
for row in ma:
    stack=[]
    for i in range(n):
        h[i]=h[i]+1 if row[i]==0 else 0
        while stack and h[stack[-1]]>h[i]:
            y=h[stack.pop()]
            w=i if not stack else i-stack[-1]-1
            ans=max(ans,y*w)
        stack.append(i)
    while stack:
        y=h[stack.pop()]
        w=n if not stack else n-stack[-1]-1
        ans=max(ans,y*w)
print(ans)
```

#### 求逆序对数/归并排序

```python
from bisect import *
a=[]
rev=0
for _ in range(n):
    num=int(input())
    rev+=bisect_left(a,num)
    insort_left(a,num)
ans=n*(n-1)//2-rev
```

```python
def merge_sort(a):
    if len(a)<=1:
        return a,0
    mid=len(a)//2
    l,l_cnt=merge_sort(a[:mid])
    r,r_cnt=merge_sort(a[mid:])
    merged,merge_cnt=merge(l,r)
    return merged,l_cnt+r_cnt+merge_cnt
def merge(l,r):
    merged=[]
    l_idx,r_idx=0,0
    inverse_cnt=0
    while l_idx<len(l) and r_idx<len(r):
        if l[l_idx]<=r[r_idx]:
            merged.append(l[l_idx])
            l_idx+=1
        else:
            merged.append(r[r_idx])
            r_idx+=1
            inverse_cnt+=len(l)-l_idx
    merged.extend(l[l_idx:])
    merged.extend(r[r_idx:])
    return merged,inverse_cnt
```



### 树

#### 根据前中序得后序、根据中后序得前序

```python
def postorder(preorder,inorder):
    if not preorder:
        return ''
    root=preorder[0]
    idx=inorder.index(root)
    left=postorder(preorder[1:idx+1],inorder[:idx])
    right=postorder(preorder[idx+1:],inorder[idx+1:])
    return left+right+root
```

```python
def preorder(inorder,postorder):
    if not inorder:
        return ''
    root=postorder[-1]
    idx=inorder.index(root)
    left=preorder(inorder[:idx],postorder[:idx])
    right=preorder(inorder[idx+1:],postorder[idx:-1])
    return root+left+right
```

#### 层次遍历

```python
from collections import deque
def levelorder(root):
    if not root:
        return ""
    q=deque([root])  
    res=""
    while q:
        node=q.popleft()  
        res+=node.val  
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res
```

#### 解析括号嵌套表达式

```python
def parse(s):
    node=Node(s[0])
    if len(s)==1:
        return node
    s=s[2:-1]; t=0; last=-1
    for i in range(len(s)):
        if s[i]=='(': t+=1
        elif s[i]==')': t-=1
        elif s[i]==',' and t==0:
            node.children.append(parse(s[last+1:i]))
            last=i
    node.children.append(parse(s[last+1:]))
    return node
```

#### 二叉搜索树的构建

```python
def insert(root,num):
    if not root:
        return Node(num)
    if num<root.val:
        root.left=insert(root.left,num)
    else:
        root.right=insert(root.right,num)
    return root
```
#### AVLTree
```python
class Note:
    def __init__(self,value):
        self.value=value
        self.lson=self.rson=None
        self.height=1

class AVLTree:
    def getHeight(self,note):
        if not note:return 0
        return note.height
    def getBalance(self,note):
        if not note:return 0
        return self.getHeight(note.lson)-self.getHeight(note.rson)
    def l_Rotate(self,z):
        y=z.rson;t=y.lson
        y.lson=z;z.rson=t
        z.height=1+max(self.getHeight(z.lson),self.getHeight(z.rson))
        y.height=1+max(self.getHeight(y.lson),self.getHeight(y.rson))
        return y
    def r_Rotate(self,y):
        x=y.lson;t=x.rson
        x.rson=y;y.lson=t
        y.height=1+max(self.getHeight(y.lson),self.getHeight(y.rson))
        x.height=1+max(self.getHeight(x.lson),self.getHeight(x.rson))
        return x
    def insert(self,root,value):
        if not root:return Note(value)
        elif value<root.value:root.lson=self.insert(root.lson,value)
        else:root.rson=self.insert(root.rson,value)
        root.height=1+max(self.getHeight(root.lson),self.getHeight(root.rson))
        balance=self.getBalance(root)
        if balance > 1 and value < root.lson.value:
                return self.r_Rotate(root)
        if balance < -1 and value > root.rson.value:
                return self.l_Rotate(root)
        if balance > 1 and value > root.lson.value:
                root.lson=self.l_Rotate(root.lson)
                return self.r_Rotate(root)
        if balance < -1 and value < root.rson.value:
                root.rson=self.r_Rotate(root.rson)
                return self.l_Rotate(root)
        return root
    def dfs(self,root):
        if not root:
            return []
        return [root.value]+self.dfs(root.lson)+self.dfs(root.rson)


n=int(input())
nums=list(map(int,input().split()))
tree=AVLTree();root=None
for i in nums:root=tree.insert(root,i)
print(*tree.dfs(root))
```

#### 并查集

```python
def find(i,j):
    if j[i] == i:return i
    else:j[i]=find(j[i],j);return j[i]
def merge(i,j,k):
    a=find(i,k);b=find(j,k)
    if a != b:k[a]=b

case=0
while True:
    n,m=map(int,input().split())
    ans=0
    if n==m==0:exit()
    case+=1
    list_=[i for i in range(n+1)]
    for _ in range(m):
        a,b=map(int,input().split())
        merge(a,b,list_)
    for i in range(1,n+1):
        if list_[i] == i:ans+=1
    print(f'Case {case}: {ans}')
```

#### 字典树的构建

```python
# 电话号码
class note:
    def __init__(self,val):
        self.val=val
        self.son=[]
class Trie:
    def __init__(self):
        self.root=note('X')
        self.leaves=0
    def add(self,string):
        string='X'+string
        t=self.root;i=1
        while i < len(string):
            flag=0
            if t.son:
                for j in t.son:
                    if j.val == string[i]:
                        t=j;i+=1;flag=1;break
            if not flag:
                t.son.append(s:=note(string[i]))
                t=s;i+=1
    def dfs(self,t):
        if not t.son:self.leaves+=1;return
        for i in t.son:self.dfs(i)
    def count(self):
        self.dfs(self.root)
        return self.leaves

for _ in range(int(input())):
    n=int(input())
    trie=Trie()
    for _ in range(n):
        trie.add(input())
    print('YES' if n == trie.count() else 'NO')
```
#### Huffman
```python
class Note:
    def __init__(self,stri,freq):
        self.stri=stri;self.freq=freq
        self.father=self.lson=self.rson=None
    def __lt__(self, other):
        if self.freq!=other.freq:return self.freq<other.freq
        else:return self.stri<other.stri
class HuffmanTree:
    def __init__(self,notes):
        self.tree=[Note([note[0]],int(note[1])) for note in notes]
        while len(self.tree) >1:self.integrate()
        self.dict = {};self.dfs(self.tree[0], '')
    def integrate(self):
        self.tree.sort()
        a=self.tree.pop(0);b=self.tree.pop(0)
        c=Note(sorted(a.stri+b.stri),a.freq+b.freq)
        a.father=b.father=c;c.lson=a;c.rson=b
        self.tree.append(c)
    def dfs(self,t,path):
        if len(t.stri) == 1:
            self.dict[t.stri[0]]=path;return
        if t.lson:self.dfs(t.lson,path+'0')
        if t.rson:self.dfs(t.rson,path+'1')
    def raw_to_code(self,raw):
        return ''.join([self.dict[i] for i in raw])
    def code_to_raw(self,code):
        raw='';i=0;t=self.tree[0]
        while i<len(code):
            if code[i] == '0':
                t=t.lson;i+=1
            elif code[i] == '1':
                t=t.rson;i+=1
            if len(t.stri) == 1:
                raw+=t.stri[0];t=self.tree[0]
        return raw

l=[list(input().split()) for _ in range(int(input()))]
HT=HuffmanTree(l)
#print(HT.dict)
try:
    while True:
        inp=input()
        if inp[0].isalpha():print(HT.raw_to_code(inp))
        else:print(HT.code_to_raw(inp))
except:exit()
```
#### 单调栈
```python
n=int(input())
ans=[0]*n;stack=[]
num=list(map(int,input().split()))
for i in range(n-1,-1,-1):
    while stack and num[stack[-1]] <= num[i]:
        stack.pop()
    ans[i]=stack[-1]+1 if stack else 0
    stack.append(i)
print(*ans)
```
#### 手搓堆
```python
inf=float('inf')
class note:
    def __init__(self,value):
        self.value=value
        # self.father=None
        # self.lson=None
        # self.rson=None       #完全二叉树可以通过序号寻找父子节点
    def __lt__(self, other):
        return self.value<other.value
    def __le__(self, other):
        return self.value<=other.value

class BinHeap:
    def __init__(self):
        self.notes=[0]
        self.length=0
    def swap(self,i,j):
        self.notes[i],self.notes[j]=self.notes[j],self.notes[i]
    def shiftdown(self,index):
        t=self.notes[index]
        a=self.notes[index*2] if self.length>=index*2 else note(inf)
        b=self.notes[index*2+1] if self.length>=index*2+1 else note(inf)
        if a<=b and a<t:
            self.swap(index,index*2);self.shiftdown(index*2)
        elif b<=a and b<t:
            self.swap(index,index*2+1);self.shiftdown(index*2+1)
        else:return
    def shiftup(self,index):
        if index == 1:return
        t=self.notes[index]
        if t<self.notes[index//2]:
            self.swap(index,index//2);self.shiftup(index//2)
        else:return
    def pop(self):
        mini=self.notes[1].value
        t=self.notes.pop();self.length-=1
        if self.length:
            self.notes[1]=t;self.shiftdown(1)
        return mini
    def append(self,num):
        self.notes.append(note(num))
        self.length+=1;self.shiftup(self.length)

heap=BinHeap()
for _ in range(int(input())):
    inp=input()
    if inp[0] == '1':heap.append(int(inp[2:]))
    else:print(heap.pop())
```

### 图


#### 棋盘问题（回溯法）

```python
def dfs(row, k):
    if k == 0:
        return 1
    if row == n:
        return 0
    count = 0
    for col in range(n):
        if board[row][col] == '#' and not col_occupied[col]:
            col_occupied[col] = True
            count += dfs(row + 1, k - 1)
            col_occupied[col] = False
    count += dfs(row + 1, k)
    return count
col_occupied = [False] * n
print(dfs(0, k))
```
```python
# 八皇后
ans = [[0] * 8 for _ in range(92)]
row = [0] * 8
order = 0
def set_queen(i):
    global order
    if i == 8:
        for l in range(8):
            ans[order][l] = str(row[l])
        order += 1
        return
    for j in range(1, 9):
        flag = 1
        for k in range(i):
            if row[k] == j or abs(k - i) == abs(row[k] - j):
                flag = 0
                break
        if flag:
            row[i] = j
            set_queen(i + 1)
set_queen(0)
for _ in range(int(input())):
    print(''.join(ans[int(input()) - 1]))
```
#### dijkstra

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

def shortest_path(graph, start, end):
    distances = dijkstra(graph, start)
    path = [end]
    while end != start:
        for neighbor, weight in graph[end].items():
            if distances[end] == distances[neighbor] + weight:
                path.append(neighbor)
                end = neighbor
                break
    path.reverse()
    return path

if __name__ == "__main__":
    # 读取输入
    P = int(input())
    locations = [input() for _ in range(P)]
    Q = int(input())
    roads = {}
    for _ in range(Q):
        location1, location2, distance = input().split()
        distance = int(distance)
        if location1 not in roads:
            roads[location1] = {}
        if location2 not in roads:
            roads[location2] = {}
        roads[location1][location2] = distance
        roads[location2][location1] = distance
    R = int(input())
    routes = [input().split() for _ in range(R)]

    # 构建图
    graph = {location: {} for location in locations}
    for location1 in roads:
        for location2 in roads[location1]:
            graph[location1][location2] = roads[location1][location2]

    # 计算最短路径并输出结果
    for route in routes:
        start, end = route
        path = shortest_path(graph, start, end)
        result = []
        for i in range(len(path) - 1):
            result.append(path[i] + "->(" + str(roads[path[i]][path[i+1]]) + ")->")
        result.append(path[-1])
        print("".join(result))
```
```python
# **Bellman-Ford算法**：Bellman-Ford算法用于解决单源最短路径问题，与Dijkstra算法不同，它可以处理带有负权边的图。算法的基本思想是通过松弛操作逐步更新节点的最短路径估计值，直到收敛到最终结果。具体步骤如下：
# 
# - 初始化一个距离数组，用于记录源节点到所有其他节点的最短距离。初始时，源节点的距离为0，其他节点的距离为无穷大。
# - 进行V-1次循环（V是图中的节点数），每次循环对所有边进行松弛操作。如果从节点u到节点v的路径经过节点u的距离加上边(u, v)的权重比当前已知的从源节点到节点v的最短路径更短，则更新最短路径。
# - 检查是否存在负权回路。如果在V-1次循环后，仍然可以通过松弛操作更新最短路径，则说明存在负权回路，因此无法确定最短路径。
# 
# Bellman-Ford算法的时间复杂度为O(V*E)，其中V是图中的节点数，E是图中的边数。

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    def bellman_ford(self, src):
        # 初始化距离数组，表示从源点到各个顶点的最短距离
        dist = [float('inf')] * self.V
        dist[src] = 0

        # 迭代 V-1 次，每次更新所有边
        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        # 检测负权环
        for u, v, w in self.graph:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return "Graph contains negative weight cycle"

        return dist

# 测试代码
g = Graph(5)
g.add_edge(0, 1, -1)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 2)
g.add_edge(1, 4, 2)
g.add_edge(3, 2, 5)
g.add_edge(3, 1, 1)
g.add_edge(4, 3, -3)

src = 0
distances = g.bellman_ford(src)
print("最短路径距离：")
for i in range(len(distances)):
    print(f"从源点 {src} 到顶点 {i} 的最短距离为：{distances[i]}")
```

```python
# Floyd_Warshall
def Floyd_Warshall(gragh):
    for l1 in roads:
        for l2 in roads:
            for l3 in roads:
                if gragh[l2][l3]>gragh[l2][l1]+gragh[l1][l3]:
                    gragh[l2][l3] = gragh[l2][l1] + gragh[l1][l3]

def shortest_path(graph, start, end):
    # distances = dijkstra(graph, start)
    Floyd_Warshall(graph)
    path = [end]
    while end != start:
        for neighbor, weight in roads[end].items():
            if graph[start][end] == graph[start][neighbor] + weight:
                path.append(neighbor)
                end = neighbor
                break
    path.reverse()
    return path
```
#### kruskal

```python
#Kruskal

#边类
class Edge:
    def __init__(self,u,v,val):
        self.u=u;self.v=v
        self.val=val

#并查集
def search(i):
    if father[i] == i:return i
    else:
        father[i]=search(father[i])
        return father[i]
def merge(i,j):
    f1=search(i);f2=search(j)
    if f1 != f2:
        father[f2]=f1;return True
    return False

#初始化
n,m=map(int,input().split())
count=0;ans=0
father=[i for i in range(n+1)]
edges=[Edge(*map(int,input().split())) for _ in range(m)]
edges.sort(key=lambda x:x.val)
for edge in edges:
    if merge(edge.u,edge.v):
        count+=1;ans+=edge.val
    if count == n-1:break
if count != n-1:print('orz')
else:print(ans)
```

#### prim

```python
#Prim

#初始化
n,m=map(int,input().split())
edges={i:{} for i in range(1,n+1)};count=0;ans=0
#这里是考虑了输入中每两个点间可能有多条边的情况
for _ in range(m):
    u,v,val=map(int,input().split())
    if u not in edges[v].keys():edges[v][u]=val
    elif val < edges[v][u]:edges[v][u]=val
    if v not in edges[u].keys():edges[u][v]=val
    elif val < edges[u][v]:edges[u][v]=val
dis=[float('inf') for _ in range(n+1)]
book=[0 for _ in range(n+1)]
for to,dist in edges[1].items():dis[to]=dist
book[1]=1
while True:
    mini=float('inf');flag=1
    for i in range(1,n+1):
        if book[i] == 0 and dis[i] < mini:
            flag=0;mini=dis[i];j=i
    if flag:break
    book[j]=1;count+=1;ans+=dis[j]
    for to,dist in edges[j].items():
        if book[to] == 0 and dis[to] > dist:
            dis[to]=dist
if count != n-1:print('orz')
else:print(ans)
```
```python
# 堆优化Prim
from heapq import heappop, heappush
def prim(matrix):
    ans=0
    pq,visited=[(0,0)],[False for _ in range(N)]
    while pq:
        c,cur=heappop(pq)
        if visited[cur]:continue
        visited[cur]=True
        ans+=c
        for i in range(N):
            if not visited[i] and matrix[cur][i]!=0:
                heappush(pq,(matrix[cur][i],i))
    return ans

while True:
    try:
        N=int(input())
        matrix=[list(map(int,input().split())) for _ in range(N)]
        print(prim(matrix))
    except:break
```
```python
# 无向图是否联通有无回路
def isConnected(G):  
    n = len(G)
    visited = [False for _ in range(n)]
    total = 0
    def dfs(v):
        nonlocal total
        visited[v] = True
        total += 1
        for u in G[v]:
            if not visited[u]:
                dfs(u)
    dfs(0)
    return total == n
def hasLoop(G):  
    n = len(G)
    visited = [False for _ in range(n)]
    def dfs(v, x): 
        visited[v] = True
        for u in G[v]:
            if visited[u] == True:
                if u != x:  
                    return True
            else:
                if dfs(u,v):  
                    return True
        return False
    for i in range(n):
        if not visited[i]:  
            if dfs(i, -1):
                return True
    return False
n, m = map(int, input().split())
G = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    G[u].append(v)
    G[v].append(u)
if isConnected(G):print("connected:yes")
else:print("connected:no")
if hasLoop(G):print("loop:yes")
else:print("loop:no")
```
#### 拓扑排序
```python
from collections import deque
def topo_sort(graph):
    in_degree={u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v]+=1
    q=deque([u for u in in_degree if in_degree[u]==0])
    topo_order=[]
    while q:
        u=q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v]-=1
            if in_degree[v]==0:
                q.append(v)
    if len(topo_order)!=len(graph):
        return []  
    return topo_order
```

#### 有向图是否有环

```python
def has_cycle(n, m, edges):
    # 创建邻接表
    graph = [[] for _ in range(n+1)]
    for x, y in edges:
        graph[x].append(y)
    # 深度优先搜索
    def dfs(node, visited, recursion_stack):
        visited[node] = True
        recursion_stack[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, visited, recursion_stack):
                    return True
            elif recursion_stack[neighbor]:
                return True
        recursion_stack[node] = False
        return False
    # 对每个节点进行深度优先搜索
    visited = [False] * (n+1)
    recursion_stack = [False] * (n+1)
    for i in range(1, n+1):
        if not visited[i]:
            if dfs(i, visited, recursion_stack):
                return "Yes"
    return "No"
T = int(input())  # 测试组数
for _ in range(T):
    N, M = map(int, input().split())  # 既定目标数、航线数
    edges = []
    for _ in range(M):
        x, y = map(int, input().split())  # 航线的起点和终点
        edges.append((x, y))
    result = has_cycle(N, M, edges)
    print(result)
```


### 工具

int(str,n)	将字符串`str`转换为`n`进制的整数。

for key,value in dict.items()	遍历字典的键值对。

for index,value in enumerate(list)	枚举列表，提供元素及其索引。

dict.get(key,default) 	从字典中获取键对应的值，如果键不存在，则返回默认值`default`。

list(zip(a,b))	将两个列表元素一一配对，生成元组的列表。

math.pow(m,n)	计算`m`的`n`次幂。

math.log(m,n)	计算以`n`为底的`m`的对数。

lrucache	

```py
from functools import lru_cache
@lru_cache(maxsize=None)
```

bisect

```python
import bisect
# 创建一个有序列表
sorted_list = [1, 3, 4, 4, 5, 7]
# 使用bisect_left查找插入点
position = bisect.bisect_left(sorted_list, 4)
print(position)  # 输出: 2
# 使用bisect_right查找插入点
position = bisect.bisect_right(sorted_list, 4)
print(position)  # 输出: 4
# 使用insort_left插入元素
bisect.insort_left(sorted_list, 4)
print(sorted_list)  # 输出: [1, 3, 4, 4, 4, 5, 7]
# 使用insort_right插入元素
bisect.insort_right(sorted_list, 4)
print(sorted_list)  # 输出: [1, 3, 4, 4, 4, 4, 5, 7]
```

```python
from collections import Counter
# 创建一个Counter对象
count = Counter(['apple', 'banana', 'apple', 'orange', 'banana', 'apple'])
# 输出Counter对象
print(count)  # 输出: Counter({'apple': 3, 'banana': 2, 'orange': 1})
# 访问单个元素的计数
print(count['apple'])  # 输出: 3
# 访问不存在的元素返回0
print(count['grape'])  # 输出: 0
# 添加元素
count.update(['grape', 'apple'])
print(count)  # 输出: Counter({'apple': 4, 'banana': 2, 'orange': 1, 'grape': 1})
```

permutations：全排列

```python
from itertools import permutations
# 创建一个可迭代对象的排列
perm = permutations([1, 2, 3])
# 打印所有排列
for p in perm:
    print(p)
# 输出: (1, 2, 3)，(1, 3, 2)，(2, 1, 3)，(2, 3, 1)，(3, 1, 2)，(3, 2, 1)
```

combinations：组合

```python
from itertools import combinations
# 创建一个可迭代对象的组合
comb = combinations([1, 2, 3], 2)
# 打印所有组合
for c in comb:
    print(c)
# 输出: (1, 2)，(1, 3)，(2, 3)
```

reduce：累次运算

```python
from functools import reduce
# 使用reduce计算列表元素的乘积
product = reduce(lambda x, y: x * y, [1, 2, 3, 4])
print(product)  # 输出: 24
```

product：笛卡尔积

```python
from itertools import product
# 创建两个可迭代对象的笛卡尔积
prod = product([1, 2], ['a', 'b'])
# 打印所有笛卡尔积对
for p in prod:
    print(p)
# 输出: (1, 'a')，(1, 'b')，(2, 'a')，(2, 'b')
```
