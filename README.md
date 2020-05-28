# Leetcode Code Practice Note

```diff
黑色->会 
黄色->需要熟练
! orange
-红色->不熟悉
```



22. Generate Parentheses

```diff
插空法 n 的情况 等于 在n-1的情况下 每个空 插一个‘（）’,所以可以用递归backtrack 直到n==1 return [‘()’]
```
32. Longest Valid Parentheses

```diff
stack,res=[(-1,“)”)], 0 如果遇到“）” 判断如果stack最后一个是不是（ 如果是就pop（） res=Max（res，I-stack[-1][1]）
```

48. Rotate Image 

```
先转置，for I in range(n) :for j in range(i) : A[I][j],A[j][I]=A[j][I],A[I][j],然后每row 开始前后换 可以用 x 和 ~x  ```
```

49. Group Anagrams:

```Hash table’s key can not be list can use tuple(list) as a substitute
Also str can not str.sort() but can sorted(str) ex: sorted(‘eat’)=[‘a’,’e’,’t’]
```

45. Jump Game II

```
设置cur_cover：每次cover都在延伸 因为 cur_cover =max(cur_cover,nums[I]+i) 但是当curcover =last时step才+1 次是 last=curcover （last初始值为0）

如果cur_cover  destination return step
```

54 Spiral Matrix 

```
While matrix 不停的pop 指导matrix 没了 先pop出第一行然后剩下每一行最后一个然后pop最后一行的倒叙 然后pop倒叙的每行的第一个 
```

59. Spiral Matrix II

```
造一个matrix matrix[I][j]=(i,j) 利用spiral matrix while matrix 每次pop 出位置 位置存那个该存的值
```

61: Rotate List: 

```
因为做同样的操作导致重复的结果出现，用%解决，listnode可以将首尾连接成环然后用prev=None，cur=head 找到要的node然后tail.next=None就可以了。
```

76 minimum substring

```

```

81. Search in Rotated Sorted Array II

```
Binary search 变形的 先在顺序里的找找不到就在另一边
```

88. Merge sorted array

```
从大往小merge 达到in place
```

91 decode way:

```
使用dp F(x)=F(x-1)+F(x-2) F(x)=F(x-1) F(x)=F(x-2)三种情况 x为first x digitals 每次进来一个 判断和前一个是否组成1-26里的字母
```

94 Binary tree in-order traversal:

```
Left-root-Right
Recursion: 先一直递归root.left然后append(root.val)然后递归root.right
```



96 Unique Binary Search Tree（DP）

选择一个n, 那么[1…n]都可以做root, left subtree 由比root小的组成, 有f(i-1)组合（因为比I小的有i-1个数） 右边有（n-i）个数 能组成f（n-i）个subtrees，然后相乘的到n个数后组成的tree个数

\99. Recover Binary Search Tree

先用inorder traverse 生成一个数组 然后用它和sort过的它对比，如果有不一样，那两个就需要替换。然后再进行recover （遍历所有node

）=a的node就=b 等于b的就等于a

102 Binary tree level order traversal:

用stack. Stack 	每次存每个level的node 然后每次循环pop出来 每个level的node append进res 然后判断这些node是否有下一个level的children 有的话append到level里 然后循环最后append到stack里 每次循环清空nums 和level

104 maximum depth of BT 

用deque([root,’*’]) 每次popleft()出来的是每一层所有的nodes 每个nodes BFS append进deque 每层之间用‘*’相隔每层元素，当识别到它时 count+=1

105 construct BT from preorder and inorder traversal：

递归，每次preorder 出来的是root 找到root的index 若left没有等于left 就在construct left subtree

106 construct BT from postorder and inorder traversal：

如出一辙，和105一样 但是postorder是left-right-root 用pop()而不是pop(0)每次pop出来的是上一个的右child所以先construct 右子树 然后再左 操作和105一样

111 minimum depth if binary tree:

BFS 因为是广度优先，所以先处理一个level的nodes 当处理同个level的一个node 为 leaf时 直接return （return early）就是最短的

也可以用recursion， 一直递归到None return 0 每个节点 return 其 left and right 的最小值

120:Triangle:

DP：n=len(T). 建立一个NxN 的 table DP: top—down 如果没有重叠 直接上一个加下一个 有重叠 下一个等于两个中加同一个小的那一个

123 Best Time to Buy and sell stock III

\# forward traversal, profits record the max profit 

\# by the ith day, this is the first transaction

​                                        Return by + after

\# backward traversal, max_profit records the max profit

\# after the ith day, this is the second transaction   



 \126. Word Ladder II

建立字典 {cog:dog, log log: lot. Lot:hot dog: dot hot: hit } key是value的转换

130:Surrounded Regions:

使用DFS 因为只有‘O’ 在matrix的edge的时候才会不被包围 所有只要搜索是否有‘O’在edge，如果有，将‘O’变成一个符号，然后遍历所有节点，若是特殊符号就把它变回原来的‘O’，如果是‘O’就变成X

131.palindrome partition:

1.recursion：dfs 从长度为1开始 如果是对称，则path+这个字串然后递归除了这个字串之后的字串,等到底了 回溯 进行长度为2 的字串 按照这个规律 递归下去 for ex： aaba—>.   a,a,b,a,ba,ab,aba,aa,b,a,ba,aab,aaba

2.dp

138 copy list with random pointer

先用dic 存新建的Node with random=none next=none 然后遍历所有node ，dic[node].next=dic[cur.next] dic[node].random=dic[node.random]  在字典内重组

139 word beak

Dp: dp[start]=1 然后遍历 若果 s[start:start+len(word)]==word 则这个单词的结尾+1 dp[index+1]=1 进行下一个单词的寻找和判断

\140. Word Break II

recursion: 利用s.startswith()  循环worddic 如果有word 是 现在s的开头 则进往下递归 s[len(word):] resultOfTheRest=self.helper(s[len(word):],dic,memo)

利用memo 记录，所以一次情况只用算一遍 如果还遇到相同的substring 直接返回 memo[s]的值 

for result in resultOfTheRest:result=word+' '+result res.append(result)



\142. Linked List Cycle II:

Linklist can be used in hash table

\148. Sort List

Merge sort: 先递归分到只剩下两个，然后再merge

152:maximum product subarray

用两个dp列表

一个存positive，一个存negitive, val=(current, pos[i-1]*current, neg[i-1]*current) pos[I]=max(val) neg[I]=min(val)

154 find minimum in rotated sorted array II

用binary search 其中 若nums[mid]<nums[mid-1]则直接return nums[mid] 如果等于nums[high] 则 high=high-1



156 Binary tree upside down

Swap ，recursion 先递归找到最左节点 最左节点就是新的root然后找是否最左节点有右children 如果有 rmost就再往右找 然后 进行swap root rmost.left, rmost.right = newroot, root.right,TreeNode(root.val) 然后return上一层



161 One edit distance

遍历一遍 s=s[:i]+t[i]+s[i+1:] break else:s=s[:i]+t[i]+s[I:] break return s==t or s==t[:-1]



163 missing range

当nums[i]> lower 时，res.append(str(lower)+’->’+str(nums[I]-1)). lower=nums[I]+1 I=I+1

173 BTS iterator

在 init 里 进行inorder traversal left-root—right 然后翻转 每次pop出来的都是最小的

179.Largest number

nums = map(str, nums). t=list(nums). t.sort(key=cmp_to_key(lambda a, b: int(b+a)-int(a+b)))

\187. Repeated DNA Sequences

一个 res set 一个 check set 若这条序列没在check见过 那就add到check里 若见过 就 add到res里

190:

bin()[2:].zfill(32)

198 house robber

nums=[0]+nums for i in range(2,len(nums)): nums[i]=max(nums[i-1],nums[i-2]+nums[i])

201Bitwise AND of Number range

range(m,n) count=0 while m!=n m>>=1 n>>=1 count+=1. Return m<<count

202 happy number: 建立一个set 存放每次的和 如果有和在set里 就会导致无限循环 所以 return false 如果和为1 return true

204 count primes : dp[] 先假设全是prime 从2 开始 若 2 是素数 那从2*2 开始 没过2个 dp[n]=0 (原来2，4，6，8，10都是1 因为2是素数 所以从4开始6，8，10 都变成0）然后加一循环直至从开头变0的数超过range最大

205 isomorphic string : 用数字代表字母 相同字母一样的数字 

207 Course schedule

Topological sort 1.建立 graph g={n: Node(n) for n in range(节点数)} 2 dfs 每个节点 如果一个节点是white 然后从这个节点能dfs到一个grey的节点说明有反向的 return false 一直回溯到头 return false

class Node:

  def __init__(self,x):

​    self.val=x

​    self.color='white'

​    self.next=[]

209:

用 slidingwindow 一开始start和end都在最初， end不断增加直到window里的sum大于等于target， 然后滑动， 先减去start所处的值 然后 加（end+1 ）——————滑动，若小于target，继续expanse end 若大于等于 则缩小window 直至小于target

211 add and search word: data structure design

Using trie

\212. Word Search II

利用查询书trie， 建立trie 然后进行dfs 每次dfs前 board[I][j]=“*” 在遍历他的之后board[I][j]=‘char’恢复board的样子，为之后的word做准备



215 kthlargest element in a array：https://leetcode.com/problems/kth-largest-element-in-an-array/discuss/330933/Python-heap-and-quick-sort

Random.choice() 一个pivot, Then set 3 [], lower equal larger then if len(larger)<k<=len(larger)+len(equal) return equal[0] elif return else return (recursion )

221.Maximum square

Dp: 像最长子串那样padding 0 然后 遍历所有点 把每个点当作一个square的右coner 若他周围最小是1 则 他=1+1 dp[i][j]=min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1])+1 因为只有周围都是相等的时候 square 才能增加一圈

222 count complete tree node

:binary search 如果是complete tree 最后一层肯定是从左开始， 所以建立函数leftdepth 去找最左边节点的深度，主函数中如果root 的左右leftdepth一样 说明左边是满的 右边不一定 所以走右节点 然后同样的操作，每次count+=2**depth 因为从低往上加 和从高往低加总和是一样 但一开始很难懂，设置count=0 initial

227 basic calculator 

现在原有的string 最前面+‘+’ 记录前一个ALU 如果遇到alu 则判断上一个 根据上一个alu 来append 进入最终的stack （* 或/ 要先pop 出上一个operand 在和现在的num操作）最终返回 sum（stack）

236 Lowest common ancestor of a binary tree:

Recursion： root 左右分别找 找到其中之一就return ，对于一个节点,如果它左边没有（None） 右边有（！=None） 说明都在右边 反之都在左边，return那个！=None 的值 ， 如果两边都有，则说明它就是那个common ancestor，然后backtrack到最头

237 del node from linklist:

Swap:  设立pre cur.val,q.val=q.val,cur.val  换到最后 pre.next=None 把最后一个节点弄掉

238 product of array except self:

不用divide 且O(n), two pointer high and low，right 记录 从右到high 的阶乘 left记录从左到low的阶乘 遍历一遍array left, right， 每个点会因为low 和 high 遍历两遍， 每次称的是left的阶乘或是right的阶乘 left*=num[low] right*=num[high] low+=1 high-=1

然后 设立res=[1]*len(array) res[low],res[high](其实就是每个点遍历两遍)*= left, right

240 search a 2D matrix

用最右上角的item做比较(row=0, col=len(matrix[0])-1)，如果大于item，则往下找 若小于item 则往左找

241Different way to add parentheses

Recursion 建立 helper 函数里得到left 和 right left 和 right 为 这个符号 左边 的所有组合的集合 和这个符合右边所有组合的集合

243 Shortest word distance:

字典 记录最新indx 每次循环更新最小dis

244 Shortest word distance II

因为只计算 call shortdistance 的时间 所以在init 里用dict 存好 在call function 里 O（1）查询 ，然后循环很少次得到结果

249 Group shifted string 

key=() key+=((ord(s[I+1]-ord(s[I]))+26)%26),) 用坐标来()当key (key+=(***,)) 同样的key append 进去 最后 输出dict.values

250 count univalve subtree

自上而下 递归 找 如果是leaf 直接加1 如果 是半tree(有一个child 且那个child 是 leaf) 如果child=root 则+1 如果left=right=root 则+1

253 Meeting room II

把每个会议的开始和结束分开 然后sort 当一个会议开始了 +1 当一个会议结束了-1 （如果有重叠 肯定会+1后又+1 则需要2个会议室了 然后之后结束时间到了 -1） return 最大的时候的值

254 Factor combinations 

256 paint house

 For each house 涂每一种颜色的total最小值 贪心到最后一个house 然后 在最后一个house三种颜色的总花销里找最小的

259 3sum smaller

Start, mid,end=I, I+1,k-1 (k=len(nums) I for I in range(len(mums)))

261 Graph valid tree

Method 1: union 每出来一个edge，其两个node 用 find 方法 找到他们的root 如果他们root不一样 因为这个edge 所以union 第一个为第二个的parent 最终union 所有edge 然后 find 每个节点 如果 大家的root都是同一个数 那return true

264 ugly number II

box,find,factor=[1],[0,0,0],[2,3,5]

​    for i in range(n-1):

​      t=[box[find[j]]*factor[j] for j in range(3)]

​      element=min(t)

​      box.append(element)

​      for j in range(3):

​        if box[find[j]]*factor[j]==element:

​          find[j]+=1

​    return box[-1]

\277. Find the Celebrity:

先假设那个名人是0 然后循环所有人，如果candidate认识那个人，那那个人就变成candidate，之后循环candidate 之前的人，如果candidate认识他们其中之一就return-1 如果他们不认识candidate 那也return-1 之后再循环candidate 之后的人 如果之后的人不认识candidate 也 return-1 因为在第一步candidate 已经不认识他后面的人了



279 perfect square

DP:自己的答案 每次循环box里的element  box 里 append(I**2)

BFS: queue 一开始存(n,0),然后(n-j,+1)偶此类推直到n-j =0 return +1

282:

Recursion: helper(cur_ind, cur_val,pre_val,exp)

285 Inorder successor of BST

Recursion: 如果大于target 往左找 如果小于target 往右找直到找到比target大的，如果找到底 return float(‘inf’) ,None 往上回溯 （到了一个节点）是由左节点回溯到这个节点，再向上return min（回溯上的值，root的值） 

290 word pattern

活用 map zip 用第一次出现的index来表示token 和pattern

296 Best meeting point

1D的时候，就是找median， 1D array 里 设置left 和right , I 和 j， left(代表 截止这个点有多少个人) 每走一步 d+=left*1，I+=1 相反 right 也是如此 j-=1 d+=right*1 当 I=j 时相遇 此时需要的距离最小（因为人多的话走的d就会更多 每次让人少的那一方走）

300 Longest increasing substring:

1.用bisect

304 Range sum 2D

Dp 每个点等于从顶点到这个点cover的长方形面积 然后return 通过加减乘除图形得到所要得到的长方形里的和

307 Range sum query— mutable

Binary Index Tree: 1. compute the first I elements 2 modify the specific value in array and let the 1 step in O(logn)

310:MHT:

利用graph 找leaves( leaves=[i for i in range(n) if len(g[i])<=1]) ，建立new leaves 循环leaves的每个leaf 从它的neighbor 删去它自己 如果他的neighbor 只有一个链接的时候 那他的neighbor 是新的leaf append 进newleaves里 以此类推

当达到最后一个leaf 且 它没有链接的时候 就return 当下的leaves

312 Brust Balloon

DP : Key point: reverse thinking.

Think about n balloons if k is the last one to burst, last_burn = nums[-1] * nums[k] * nums[n].

We can see that the balloons is again separated into 2 sections. The left and right section

now has well defined boundary and do not affect each other! Therefore we can use dynamic algorithm.



各种stock：

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/404998/All-in-One-O(n)-time-O(1)-space-Python-solution

用两个变量， 一个profit after buying s1 一个 profit after selling s2

一开始你没profit 所以 s1=-prices[0] s1和s2都要尽可能大 因为当 s1 改变的时候说明之前sell 后 得到的profit再买当天的进价会多出来钱此时 就可以再进行一次buy sell 如果还没之前s1大 说明 还要等 ex 1，4，5，6 不能4-1 +5-6 因为 如果卖4 profit=3

再买5的话 s1 就等于 -2 了 比之前还少 若果后面遇到任何大的price 之前的-1 总能得到比-2 多的profit



315 Count of Smaller Numbers After Self

运用Binary index tree ,(reversed（nums）） sorted（nums） bisect.bisectleft

316 Remove Duplicate Letters  

用字典记录每个字母出现的最后的index 设置cur_res=[] 循环 如果没有append 如果字母小于cur_res最后一个 则比较这个较小的字母的ind和curres最后一个字母最后一次出现的index 如果index大于当前i 则pop掉 以此类推直至不会出现这样情况 curres append 当前字母

\329. Longest Increasing Path in a Matrix  

DFS+memorized DP dfs部分很简单 就是遍历每个点的时候会重复遍历一些点，所以第一次便利记录下来 然后为之后服务 之后遇到直接return 值就好



\336. Palindrome Pairs

循环每个word，然后循环 j for j in range(len(word))：

pre=w[:j] suf=w[j:]

如果pre是对称的 且suf[::-1] 在words 里 则可以组成新的对称string

如果suf是对称的 且pre[::-1] 在words 里 则也可以组成新的对称string

Note：因为 pre= ‘’ 的时候也满足pre对称 注意不能出现suf[：：-1]=word 这样会重复 同样在第二条条件适用 pre[：：-1]！=word

并且 j=len（word） 只用判断一次就够了 比如” abcd","dcba"



337 house robber III

binary tree 从底往高回溯 res(*,*) res[0]: the max money gained if rob the current house res[1] the max money gain if not rob the house 从0，0 往上回溯 return(root.val + left[1]+right[1], max(left[1],left[0])+max(right[1],right[0]))

\340. Longest Substring with At Most K Distinct Characters

Sliding window

Dict 计算dic里key的数量 

348 Design Tic Tac Toe

利用字典 每当下一子 它所在的对角线，行，列 及（row,col,row-col,row+col）+=1 字典名称由三个元素组成 player , 所在行or 列 or 对角线 但由于行列对角线的值可能重，所以在加上他们的下标来区分。



361 Bomb enemy

dp: 先以每行为单位循环这一行的所有列，如果有E enemy+=1 到最后进行propagate_row：将这一行所有是0的node ==enemy总数:(若其中遇到W 就 先propagate W 之前的 然后 enmey 初始化为0 继续循环 最后propogate)

当row操作完了 以同样的方式操作col(hint: while row>=0 and grid[row][col]!=‘W’: if grid[row][col]==‘0’: dp[row][col]+=value row-=1)

\362. Design Hit Counter

构造数据结构listnode 使用链表记录每个timestamp 的操作 且有next, pre ，count

365 Water and jug problem

利用math.gcd 求最大公因数 如果z%gcd(x,y)==0 return true else false 如果z>x+y return False 还要考虑都是0的情况

\373. Find K Pairs with Smallest Sums

利用heapq 可以多元素heap ex heapq.push(heap,[1,2,3,4,5,6])按优先级排序 

377 combination sum IV

Dp dp[I] 为target=I 有多少种组合 dp[0]=1 因为肯定从dp(min(nums)) 开始 有数 等于1 dp[i]=sum(dp[i-j] for j in nums if i>=j)

378 kth element in sorted matrix

heap=[]  因为是找最大值

heapq.heappush(heap, nextVal) 把数字push 进入heap 最顶是minheap

heapq.heappushpop(heap, nextVal) pop minheap 然后push 新的值 （因为找的是第k大的 单用到heap 所以全取负数 可以吧最大的值 放到最顶上 最上面如果是-15，-13来了 把-15pop -13换到最顶上）





\384. Shuffle an Array

Copy 一个list nums 要用 nums[:]



390 Elimination Game]

找规律 双数和单数情况也不相同 step head remianing



\395. Longest Substring with At Least K Repeating Characters

Recursion：

用dict 存每个字母出现的次数 当次数小于k append 进 stop中 ， 循环s start=0 从start开始 如果遇到stop里的字母 就append 进入valid_s里 （但valid 里不是最终答案 因为stop word 可能截断出现次数>k的字母 所以要持续recursion valid_s里的substring 直至整个substring 没有stopword）



\399. Evaluate Division

BFS: 首先建立graph 两个数之间相除就相当于edge {n:{d:val}}  在主函数中循环query， 如果两个n,d 都在 graph 里 则BFS(queue 里原来是（a,1）因为a/b=3 所以通过pop a 然后循环a的adj append 进入queue(b,3)然后再通过b找c NOTE：graph 一定要设立visited 这样避免死循环 b找c 之后又找b。。。。return res 如果res=None return -1.0)



\403. Frog Jump:

Dfs 每一个情况 1种情况3各分支 再每一个情况3个分支 直至 走到最终节点 没能往前走 return false 每一种情况 由他的分支 return 回来 3个 True or False

若果有一个True 就return True。backtrack 最头





\407. Trapping Rain Water II

使用heapq: 先把边缘的方块节点push 进入heapq 因为边缘节点不可能存水，然后while heap 每次pop 因为根据木桶效应，被包围的节点只能存它周围最矮那个节点高度的水，所以用heapq.pop出h最小的 然后ans加 然后 这个被包围的节点用过了，就push它进入heap 当作其他节点的外围，但如果这个节点比它周围都高肯定存不了水，所以push 进去的是max（h，它本身的高度），然后在heightmap里设置它为-1，因为便利过了

\410. Split Array Largest Sum

Binary search 

subarry 的sum 最大是sum（nums）最小 max（nums） 假设midpoint 就是那个值 for num in nums tmpsum+=num if tmpsum>midpoint tmpsum=sum count+=1 如果count 值小于m 说明mid值大了 反之小了 继续search

\416. Partition Equal Subset Sum  

it's the same as the 1-D DP of knapsack but replace the maximum with OR as we just want to know if the bag can contain exactly the same amount as claimed.

循环所有num dp=[1]+[0]*target ，给定特定值 要刚好装满 看他是否存在dp[s-num] 如果dp[s-num]=1 那咋此num的循环下 dp[s]=1 设置dp[0]=1因为在num里的数字dp[num-num]=dp[0]=1

\417. Pacific Atlantic Water Flow:

BFS 和 DFS 都能做 活用dfs 和bfs 不难 （两个set 可以用& 找到公共元素）

\419. Battleships in a Board  

 Since there must be '.' between any two battleships, we can count battleships by counting their top-left corner's 'X'.That is, board[i][j]=='X' and (not i or board[i-1][j]=='.') and (not j or board[i][j-1]=='.')).If a board[i][j] == 'X' but its either left or above cell is also an 'X', then it must be a part of a battleship that we have already counted by counting its top-left 'X'. So we don't count this kind of 'X'.

\421. Maximum XOR of Two Numbers in an Array

活用zip  

if L = 5, then 3 = [0, 0, 0, 1, 1], so the steps to get there are: 活用Trie 查询树 找尽可能多的相反bit 若果没有1-bit 就 按bit 往下照

\# (3 >> 4) & 1 = 0

\# (3 >> 3) & 1 = 0

\# (3 >> 2) & 1 = 0

\# (3 >> 1) & 1 = 1

\# (3 >> 0) & 1 = 1

