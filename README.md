# Leetcode Code Practice Note

```diff
 黑色 -> 已经熟练 
!橘色 -> 需要熟练
-红色 -> 不熟悉
```

4. Median of Two Sorted Arrays

```diff
-不熟悉
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        lens = len(nums1) + len(nums2)
        if lens % 2 == 1:
            return self.kthSmallest(nums1, nums2, lens//2)
        else:
            return ( self.kthSmallest(nums1, nums2, lens//2 - 1) + self.kthSmallest(nums1, nums2, lens//2) ) / 2.0
        
def kthSmallest(self, nums1, nums2, k):
        if not nums1:
        return nums2[k]
        if not nums2:
        return nums1[k]

midIdx1, midIdx2 = len(nums1)//2, len(nums2)//2
midVal1, midVal2 = nums1[midIdx1], nums2[midIdx2]

# when k is relatively large, then we can safely drop the first half that are surely smaller than the kth
# the question is where is the first half that are surely smaller than the kth?
# by comparing midVal1 and midVal2, we can find it out
# if midVal1 < midVal2, then all the vals in nums1[:midIdx1] are less than midVal2 
# also all of those vals are less than kth, we can safely drop all those vals
if k > midIdx1 + midIdx2:
if midVal1 < midVal2:   
return self.kthSmallest(nums1[midIdx1 + 1:], nums2, k - midIdx1 - 1)
else:
return self.kthSmallest(nums1, nums2[midIdx2 + 1:], k - midIdx2 - 1)

# when k is relatively small, then we can safely drop the second half that are surely larger than the kth
# the question is where is the second half that are surely larger then the kth?
# by comparing midVal1 and midVal2, we can find it out
# if midVal1 > midVal2, then all the vals in nums1[midIdx1:] are larger than midVal2
# also all of those vals are larger than kth, we can safely drop all those vals
else:
if midVal1 > midVal2:
return self.kthSmallest(nums1[:midIdx1], nums2, k)
else:
return self.kthSmallest(nums1, nums2[:midIdx2], k)
```

5. Longest Palindromic Substring

```
!O(n)：设置maxL和maxS 遍历一遍,每次遍历判断两次，每次判断i-maxL到i 是否对称 和 i-maxL-1到i是否对称 更新maxL 和 maxS
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

```diff
先转置，for I in range(n) :for j in range(i) : A[I][j],A[j][I]=A[j][I],A[I][j],然后每row 开始前后换 可以用 x 和 ~x  ```
```

49. Group Anagrams:

```
Also str can not str.sort() but can sorted(str) ex: sorted(‘eat’)=[‘a’,’e’,’t’]
```

45. Jump Game II

```
设置cur_cover：每次cover都在延伸 因为 cur_cover =max(cur_cover,nums[I]+i) 但是当curcover =last时step才+1 次是 last=curcover （last初始值为0）

如果cur_cover  destination return step
```

54. Spiral Matrix 

```diff
While matrix 不停的pop 指导matrix 没了 先pop出第一行然后剩下每一行最后一个然后pop最后一行的倒叙 然后pop倒叙的每行的第一个 
```

59. Spiral Matrix II

```
造一个 matrix matrix[i][j]=(i,j) 利用spiral matrix while matrix 每次pop 出位置 位置存那个该存的值
```

61. Rotate List: 

```
因为做同样的操作导致重复的结果出现，用%解决，listnode可以将首尾连接成环然后用prev=None，cur=head 找到要的node然后tail.next=None就可以了。
```

76. minimum substring

```

```

81. Search in Rotated Sorted Array II

```
Binary search 变形的 先在顺序里的找找不到就在另一边
```

85. Maximal Rectangle

```diff
-Every row in the matrix is viewed as the ground with some buildings on it. 
The building height is the count of consecutive 1s from that row to above rows. 
for every row height 记录了当前row下 一些building的高度，
循环height的长度（trick height的长度为row长度+1 因为最后一个是0）
利用stack 如果当前building的高度小于stk[-1]的高度 说明 有了断层，
断层的高度h就是height[stk.pop()]
宽度就是当前i-1-stk[-1] 这里的stk[-1] 存放着>=h 的building的index，算出长方形面积。 直至当前i的高度不在小于stk[-1]的高度。

```



88. Merge sorted array

```
从大往小merge 达到in plac
```

91. decode way:

```
使用dp F(x)=F(x-1)+F(x-2) F(x)=F(x-1) F(x)=F(x-2)三种情况 x为first x digitals 每次进来一个 判断和前一个是否组成1-26里的字母
```

94. Binary tree in-order traversal:

```
Left-root-Right
Recursion: 先一直递归root.left然后append(root.val)然后递归root.right
```

96. Unique Binary Search Tree（DP）

```
选择一个n, 那么[1…n] 都可以做root, left subtree 由比root小的组成, 有f(i-1)组合（因为比I小的有i-1个数） 右边有（n-i）个数 能组成f（n-i）个subtrees，然后相乘的到n个数后组成的tree个数
```

99. Recover Binary Search Tree

```
先用inorder traverse 生成一个数组 然后用它和sort过的它对比，如果有不一样，那两个就需要替换。然后再进行recover （遍历所有node=a的node就=b 等于b的就等于a
```

102. Binary tree level order traversal:

```
用stack. Stack 每次存每个level的node 然后每次循环pop出来 每个level的node append进res 然后判断这些node是否有下一个level的children 有的话append到level里 然后循环最后append到stack里 每次循环清空nums 和level
```

104. maximum depth of BT 

```
用deque([root,’*’]) 每次popleft()出来的是每一层所有的nodes 每个nodes BFS append进deque 每层之间用‘*’相隔每层元素，当识别到它时 count+=1
```

105. construct BT from preorder and inorder traversal：

```
递归，每次preorder 出来的是root 找到root的index 若left没有等于left 就在construct left subtree
```

106. construct BT from postorder and inorder traversal：

```
如出一辙，和105一样 但是postorder是left-right-root 用pop()而不是pop(0)每次pop出来的是上一个的右child所以先construct 右子树 然后再左 操作和105一样
```

111. minimum depth if binary tree:

```
BFS 因为是广度优先，所以先处理一个level的nodes 当处理同个level的一个node 为 leaf时 直接return （return early）就是最短的

也可以用recursion， 一直递归到None return 0 每个节点 return 其 left and right 的最小值
```

120. Triangle:

```
DP：n=len(T). 建立一个NxN 的 table DP: top—down 如果没有重叠 直接上一个加下一个 有重叠 下一个等于两个中加同一个小的那一个
```

123. Best Time to Buy and sell stock III

```
forward traversal, profits record the max profit 
by the ith day, this is the first transaction 
Return by + after
backward traversal, max_profit records the max profit
after the ith day, this is the second transaction   
```

126. Word Ladder II

```
建立字典 {cog:dog, log log: lot. Lot:hot dog: dot hot: hit } key是value的转换
```

130. Surrounded Regions:

```
使用DFS 因为只有‘O’ 在matrix的edge的时候才会不被包围 所有只要搜索是否有‘O’在edge，如果有，将‘O’变成一个符号，然后遍历所有节点，若是特殊符号就把它变回原来的‘O’，如果是‘O’就变成X
```

131. palindrome partition:

```
1.recursion：dfs 从长度为1开始 如果是对称，则path+这个字串然后递归除了这个字串之后的字串,等到底了 回溯 进行长度为2 的字串 按照这个规律 递归下去 for ex： aaba—>.   a,a,b,a,ba,ab,aba,aa,b,a,ba,aab,aaba
2.dp
```

138. copy list with random pointer

```
先用dic 存新建的Node with random=none next=none 然后遍历所有node ，dic[node].next=dic[cur.next] dic[node].random=dic[node.random]  在字典内重组
```

139. word beak

```
Dp: dp[start]=1 然后遍历 若果 s[start:start+len(word)]==word 则这个单词的结尾+1 dp[index+1]=1 进行下一个单词的寻找和判断
```

140. Word Break II

```
recursion: 利用s.startswith()  循环worddic 如果有word 是 现在s的开头 则进往下递归 s[len(word):] resultOfTheRest=self.helper(s[len(word):],dic,memo)

利用memo 记录，所以一次情况只用算一遍 如果还遇到相同的substring 直接返回 memo[s]的值 

for result in resultOfTheRest:result=word+' '+result res.append(result)
```

142. Linked List Cycle II:

```
Linklist can be used in hash table
```

148. Sort List

```
Merge sort: 先递归分到只剩下两个，然后再merge
```

152. maximum product subarray

```
用两个dp列表

一个存positive，一个存negitive, val=(current, pos[i-1]*current, neg[i-1]*current) pos[I]=max(val) neg[I]=min(val)
```

154. find minimum in rotated sorted array II

```
用binary search 其中 若nums[mid]<nums[mid-1]则直接return nums[mid] 如果等于nums[high] 则 high=high-1
```

156. Binary tree upside down

```
Swap ，recursion 先递归找到最左节点 最左节点就是新的root然后找是否最左节点有右children 如果有 rmost就再往右找 然后 进行swap root rmost.left, rmost.right = newroot, root.right,TreeNode(root.val) 然后return上一层
```

161. One edit distance

```
遍历一遍 s=s[:i]+t[i]+s[i+1:] break else:s=s[:i]+t[i]+s[I:] break return s==t or s==t[:-1]
```

163. missing range

```
当nums[i]> lower 时，res.append(str(lower)+’->’+str(nums[I]-1)). lower=nums[I]+1 I=I+1
```

173. BTS iterator

```
在 init 里 进行inorder traversal left-root—right 然后翻转 每次pop出来的都是最小的
```

179. Largest number

```
nums = map(str, nums). t=list(nums). t.sort(key=cmp_to_key(lambda a, b: int(b+a)-int(a+b)))
```

186. Reverse Words in a String II

```diff
!In-place: 首先，先把整个string reverse；
然后，遍历找到每个单词的首尾（通过空格），然后再用left=首，right=尾 互换，和第一步一样，最后得到结果
```



187. Repeated DNA Sequences

```
一个 res set 一个 check set 若这条序列没在check见过 那就add到check里 若见过 就 add到res里
```

190. 

```
bin()[2:].zfill(32)
```

198. house robber

```
nums=[0]+nums for i in range(2,len(nums)): nums[i]=max(nums[i-1],nums[i-2]+nums[i])
```



201. Bitwise AND of Number range

```
range(m,n) count=0 while m!=n m>>=1 n>>=1 count+=1. Return m<<count
```

202. happy number

     ```
     建立一个set 存放每次的和 如果有和在set里 就会导致无限循环 所以 return false 如果和为1 return true
     ```

203.  count primes

```
dp[] 先假设全是prime 从2 开始 若 2 是素数 那从2*2 开始 没过2个 dp[n]=0 (原来2，4，6，8，10都是1 因为2是素数 所以从4开始6，8，10 都变成0）然后加一循环直至从开头变0的数超过range最大
```

205. isomorphic string

```
用数字代表字母 相同字母一样的数字 
```

207. Course schedule

```
Topological sort 1.建立 graph g={n: Node(n) for n in range(节点数)} 2 dfs 每个节点 如果一个节点是white 然后从这个节点能dfs到一个grey的节点说明有反向的 return false 一直回溯到头 return false

class Node:

  def __init__(self,x):

​    self.val=x

​    self.color='white'

​    self.next=[]
```

209. 

```
用 slidingwindow 一开始start和end都在最初， end不断增加直到window里的sum大于等于target， 然后滑动， 先减去start所处的值 然后 加（end+1 ）——————滑动，若小于target，继续expanse end 若大于等于 则缩小window 直至小于target
```

211. add and search word: data structure design

```
Using trie
```



212. Word Search II

```
利用查询书trie， 建立trie 然后进行dfs 每次dfs前 board[I][j]=“*” 在遍历他的之后board[I][j]=‘char’恢复board的样子，为之后的word做准备
```

215. kthlargest element in a array

```
Random.choice() 一个pivot, Then set 3 [], lower equal larger then if len(larger)<k<=len(larger)+len(equal) return equal[0] elif return else return (recursion )

https://leetcode.com/problems/kth-largest-element-in-an-array/discuss/330933/Python-heap-and-quick-sort
```

221. Maximum square

```
Dp: 像最长子串那样padding 0 然后 遍历所有点 把每个点当作一个square的右coner 若他周围最小是1 则 他=1+1 dp[i][j]=min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1])+1 因为只有周围都是相等的时候 square 才能增加一圈
```

222. count complete tree node

```
binary search 如果是complete tree 最后一层肯定是从左开始， 所以建立函数leftdepth 去找最左边节点的深度，主函数中如果root 的左右leftdepth一样 说明左边是满的 右边不一定 所以走右节点 然后同样的操作，每次count+=2**depth 因为从低往上加 和从高往低加总和是一样 但一开始很难懂，设置count=0 initial
```

227. basic calculator 

```
现在原有的string 最前面+‘+’ 记录前一个ALU 如果遇到alu 则判断上一个 根据上一个alu 来append 进入最终的stack （* 或/ 要先pop 出上一个operand 在和现在的num操作）最终返回 sum（stack）
```

233. Number of Digit One

```
找规律，垃圾题
class Solution:
    def countDigitOne(self, n):
        if n <= 0:
            return 0
        q, x, ans = n, 1, 0
        while q > 0:
            digit = q % 10
            q //= 10
            ans += q * x
            print(ans)
            if digit == 1:
                ans += n % x + 1
            elif digit > 1:
                ans += x
            print(ans)
            x *= 10
        return an     
```



236. Lowest common ancestor of a binary tree:

```
Recursion： root 左右分别找 找到其中之一就return ，对于一个节点,如果它左边没有（None） 右边有（！=None） 说明都在右边 反之都在左边，return那个！=None 的值 ， 如果两边都有，则说明它就是那个common ancestor，然后backtrack到最头
```

237. del node from linklist:

```
Swap:  设立pre cur.val,q.val=q.val,cur.val  换到最后 pre.next=None 把最后一个节点弄掉
```

238. product of array except self:

```
不用divide 且O(n), two pointer high and low，right 记录 从右到high 的阶乘 left记录从左到low的阶乘 遍历一遍array left, right， 每个点会因为low 和 high 遍历两遍， 每次称的是left的阶乘或是right的阶乘 left*=num[low] right*=num[high] low+=1 high-=1

然后 设立res=[1]*len(array) res[low],res[high](其实就是每个点遍历两遍)*= left, right
```

240. search a 2D matrix

```
用最右上角的item做比较(row=0, col=len(matrix[0])-1)，如果大于item，则往下找 若小于item 则往左找
```

241. Different way to add parentheses

```
Recursion 建立 helper 函数里得到left 和 right left 和 right 为 这个符号 左边 的所有组合的集合 和这个符合右边所有组合的集合
```

243. Shortest word distance:

```
字典 记录最新indx 每次循环更新最小dis
```

244. Shortest word distance II

```
因为只计算 call shortdistance 的时间 所以在init 里用dict 存好 在call function 里 O（1）查询 ，然后循环很少次得到结果
```

249. Group shifted string 

```
key=() key+=((ord(s[I+1]-ord(s[I]))+26)%26),) 用坐标来()当key (key+=(***,)) 同样的key append 进去 最后 输出dict.values
```

250. count univalve subtree

```
自上而下 递归 找 如果是leaf 直接加1 如果 是半tree(有一个child 且那个child 是 leaf) 如果child=root 则+1 如果left=right=root 则+1
```

253. Meeting room II

```
把每个会议的开始和结束分开 然后sort 当一个会议开始了 +1 当一个会议结束了-1 （如果有重叠 肯定会+1后又+1 则需要2个会议室了 然后之后结束时间到了 -1） return 最大的时候的值
```

254. Factor combinations 

256. paint house

```
For each house 涂每一种颜色的total最小值 贪心到最后一个house 然后 在最后一个house三种颜色的总花销里找最小的
```

259. 3sum smaller

```
Start, mid,end=I, I+1,k-1 (k=len(nums) I for I in range(len(mums)))
```

261. Graph valid tree

```
Method 1: union 每出来一个edge，其两个node 用 find 方法 找到他们的root 如果他们root不一样 因为这个edge 所以union 第一个为第二个的parent 最终union 所有edge 然后 find 每个节点 如果 大家的root都是同一个数 那return true
```

264. ugly number II

```
box,find,factor=[1],[0,0,0],[2,3,5]
for i in range(n-1):
	t=[box[find[j]]*factor[j] for j in range(3)]
	element=min(t)
	box.append(element)
	for j in range(3):
		if box[find[j]]*factor[j]==element:
		find[j]+=1
return box[-1]
```

277. Find the Celebrity:

```
先假设那个名人是0 然后循环所有人，如果candidate认识那个人，那那个人就变成candidate，之后循环candidate 之前的人，如果candidate认识他们其中之一就return-1 如果他们不认识candidate 那也return-1 之后再循环candidate 之后的人 如果之后的人不认识candidate 也 return-1 因为在第一步candidate 已经不认识他后面的人了
```

279. perfect square

```
DP:自己的答案 每次循环box里的element  box 里 append(I**2)

BFS: queue 一开始存(n,0),然后(n-j,+1)偶此类推直到n-j =0 return +1
```

282. 

```
Recursion: helper(cur_ind, cur_val,pre_val,exp)
```

285. Inorder successor of BST

```
Recursion: 如果大于target 往左找 如果小于target 往右找直到找到比target大的，如果找到底 return float(‘inf’) ,None 往上回溯 （到了一个节点）是由左节点回溯到这个节点，再向上return min（回溯上的值，root的值） 
```

290. word pattern

```
活用 map zip 用第一次出现的index来表示token 和pattern
```

296. Best meeting point

```
1D的时候，就是找median， 1D array 里 设置left 和right , I 和 j， left(代表 截止这个点有多少个人) 每走一步 d+=left*1，I+=1 相反 right 也是如此 j-=1 d+=right*1 当 I=j 时相遇 此时需要的距离最小（因为人多的话走的d就会更多 每次让人少的那一方走）
```

300. Longest increasing substring:

```
1.用bisect
```

304. Range sum 2D

```
Dp 每个点等于从顶点到这个点cover的长方形面积 然后return 通过加减乘除图形得到所要得到的长方形里的和
```

307. Range sum query— mutable

```
Binary Index Tree: 1. compute the first I elements 2 modify the specific value in array and let the 1 step in O(logn)
```

310. MHT

```
利用graph 找leaves( leaves=[i for i in range(n) if len(g[i])<=1]) ，建立new leaves 循环leaves的每个leaf 从它的neighbor 删去它自己 如果他的neighbor 只有一个链接的时候 那他的neighbor 是新的leaf append 进newleaves里 以此类推

当达到最后一个leaf 且 它没有链接的时候 就return 当下的leaves
```

312. Brust Balloon

```
DP : Key point: reverse thinking.

Think about n balloons if k is the last one to burst, last_burn = nums[-1] * nums[k] * nums[n].

We can see that the balloons is again separated into 2 sections. The left and right section

now has well defined boundary and do not affect each other! Therefore we can use dynamic algorithm.
```

```
各种stock：

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/404998/All-in-One-O(n)-time-O(1)-space-Python-solution

用两个变量， 一个profit after buying s1 一个 profit after selling s2

一开始你没profit 所以 s1=-prices[0] s1和s2都要尽可能大 因为当 s1 改变的时候说明之前sell 后 得到的profit再买当天的进价会多出来钱此时 就可以再进行一次buy sell 如果还没之前s1大 说明 还要等 ex 1，4，5，6 不能4-1 +5-6 因为 如果卖4 profit=3

再买5的话 s1 就等于 -2 了 比之前还少 若果后面遇到任何大的price 之前的-1 总能得到比-2 多的profit
```

315. Count of Smaller Numbers After Self

```
运用Binary index tree ,(reversed（nums）） sorted（nums） bisect.bisectleft
```

316. Remove Duplicate Letters  

```
用字典记录每个字母出现的最后的index 设置cur_res=[] 循环 如果没有append 如果字母小于cur_res最后一个 则比较这个较小的字母的ind和curres最后一个字母最后一次出现的index 如果index大于当前i 则pop掉 以此类推直至不会出现这样情况 curres append 当前字母
```

329. Longest Increasing Path in a Matrix  

```
DFS+memorized DP dfs部分很简单 就是遍历每个点的时候会重复遍历一些点，所以第一次便利记录下来 然后为之后服务 之后遇到直接return 值就好
```

336. Palindrome Pairs

```
循环每个word，然后循环 j for j in range(len(word))：

pre=w[:j] suf=w[j:]

如果pre是对称的 且suf[::-1] 在words 里 则可以组成新的对称string

如果suf是对称的 且pre[::-1] 在words 里 则也可以组成新的对称string

Note：因为 pre= ‘’ 的时候也满足pre对称 注意不能出现suf[：：-1]=word 这样会重复 同样在第二条条件适用 pre[：：-1]！=word

并且 j=len（word） 只用判断一次就够了 比如” abcd","dcba"
```

337. house robber III

```
binary tree 从底往高回溯 res(*,*) res[0]: the max money gained if rob the current house res[1] the max money gain if not rob the house 从0，0 往上回溯 return(root.val + left[1]+right[1], max(left[1],left[0])+max(right[1],right[0]))
```

340. Longest Substring with At Most K Distinct Characters

```
Sliding window

Dict 计算dic里key的数量 
```

348. Design Tic Tac Toe

```
利用字典 每当下一子 它所在的对角线，行，列 及（row,col,row-col,row+col）+=1 字典名称由三个元素组成 player , 所在行or 列 or 对角线 但由于行列对角线的值可能重，所以在加上他们的下标来区分。
```

361. Bomb enemy

```
dp: 先以每行为单位循环这一行的所有列，如果有E enemy+=1 到最后进行propagate_row：将这一行所有是0的node ==enemy总数:(若其中遇到W 就 先propagate W 之前的 然后 enmey 初始化为0 继续循环 最后propogate)

当row操作完了 以同样的方式操作col(hint: while row>=0 and grid[row][col]!=‘W’: if grid[row][col]==‘0’: dp[row][col]+=value row-=1)
```

362. Design Hit Counter

```
构造数据结构listnode 使用链表记录每个timestamp 的操作 且有next, pre ，count
```

365. Water and jug problem

```
利用math.gcd 求最大公因数 如果z%gcd(x,y)==0 return true else false 如果z>x+y return False 还要考虑都是0的情况
```

373. Find K Pairs with Smallest Sums

```
利用heapq 可以多元素heap ex heapq.push(heap,[1,2,3,4,5,6])按优先级排序 
```

377. combination sum IV

```
Dp dp[I] 为target=I 有多少种组合 dp[0]=1 因为肯定从dp(min(nums)) 开始 有数 等于1 dp[i]=sum(dp[i-j] for j in nums if i>=j)
```

378. kth element in sorted matrix

```
heap=[]  因为是找最大值

heapq.heappush(heap, nextVal) 把数字push 进入heap 最顶是minheap

heapq.heappushpop(heap, nextVal) pop minheap 然后push 新的值 （因为找的是第k大的 单用到heap 所以全取负数 可以吧最大的值 放到最顶上 最上面如果是-15，-13来了 把-15pop -13换到最顶上
```

384. Shuffle an Array

```
Copy 一个list nums 要用 nums[:]
```

390. Elimination Game]

```
找规律 双数和单数情况也不相同 step head remianing
```

395. Longest Substring with At Least K Repeating Characters

```
Recursion：

用dict 存每个字母出现的次数 当次数小于k append 进 stop中 ， 循环s start=0 从start开始 如果遇到stop里的字母 就append 进入valid_s里 （但valid 里不是最终答案 因为stop word 可能截断出现次数>k的字母 所以要持续recursion valid_s里的substring 直至整个substring 没有stopword）
```

399. Evaluate Division

```
BFS: 首先建立graph 两个数之间相除就相当于edge {n:{d:val}}  在主函数中循环query， 如果两个n,d 都在 graph 里 则BFS(queue 里原来是（a,1）因为a/b=3 所以通过pop a 然后循环a的adj append 进入queue(b,3)然后再通过b找c NOTE：graph 一定要设立visited 这样避免死循环 b找c 之后又找b。。。。return res 如果res=None return -1.0)
```

403. Frog Jump:

```
Dfs 每一个情况 1种情况3各分支 再每一个情况3个分支 直至 走到最终节点 没能往前走 return false 每一种情况 由他的分支 return 回来 3个 True or False

若果有一个True 就return True。backtrack 最头
```

407. Trapping Rain Water II

```
使用heapq: 先把边缘的方块节点push 进入heapq 因为边缘节点不可能存水，然后while heap 每次pop 因为根据木桶效应，被包围的节点只能存它周围最矮那个节点高度的水，所以用heapq.pop出h最小的 然后ans加 然后 这个被包围的节点用过了，就push它进入heap 当作其他节点的外围，但如果这个节点比它周围都高肯定存不了水，所以push 进去的是max（h，它本身的高度），然后在heightmap里设置它为-1，因为便利过了
```

410. Split Array Largest Sum

```
Binary search 

subarry 的sum 最大是sum（nums）最小 max（nums） 假设midpoint 就是那个值 for num in nums tmpsum+=num if tmpsum>midpoint tmpsum=sum count+=1 如果count 值小于m 说明mid值大了 反之小了 继续search
```

416. Partition Equal Subset Sum  

```
it's the same as the 1-D DP of knapsack but replace the maximum with OR as we just want to know if the bag can contain exactly the same amount as claimed.

循环所有num dp=[1]+[0]*target ，给定特定值 要刚好装满 看他是否存在dp[s-num] 如果dp[s-num]=1 那咋此num的循环下 dp[s]=1 设置dp[0]=1因为在num里的数字dp[num-num]=dp[0]=1
```

417. Pacific Atlantic Water Flow:

```
BFS 和 DFS 都能做 活用dfs 和bfs 不难 （两个set 可以用& 找到公共元素）
```

419. Battleships in a Board  

```
 Since there must be '.' between any two battleships, we can count battleships by counting their top-left corner's 'X'.That is, board[i][j]=='X' and (not i or board[i-1][j]=='.') and (not j or board[i][j-1]=='.')).If a board[i][j] == 'X' but its either left or above cell is also an 'X', then it must be a part of a battleship that we have already counted by counting its top-left 'X'. So we don't count this kind of 'X'.
```

421. Maximum XOR of Two Numbers in an Array

```
活用zip  

if L = 5, then 3 = [0, 0, 0, 1, 1], so the steps to get there are: 活用Trie 查询树 找尽可能多的相反bit 若果没有1-bit 就 按bit 往下照
# (3 >> 4) & 1 = 0
# (3 >> 3) & 1 = 0
# (3 >> 2) & 1 = 0
# (3 >> 1) & 1 = 1
# (3 >> 0) & 1 = 1
```

424. Longest Repeating Character Replacement  

```
silding window：start end (非常重要 要记录window里出现最多次的字母次数)
```

426. Convert Binary Search Tree to Sorted Doubly Linked List

```
设置lhead ltail rhead rtail   cur.left=ltail head=lhead cur.right=rhead, tail=rtail

递归
```

437. Path Sum III

```
活用dict dict存每次递增的数值dfs到某一个点的值-target 在字典里的数 以这个node为结束点 总和为target的path的数量
```

438. Find All Anagrams in a String  

```
有字母string的时候 活用[0]*26 ord ，滑动窗口，每次只改变2个值，移去之前的开头，和窗口末尾加入一个
```

442. Find All Duplicates in an Array

```
因为0<a[I]<=n 所以可以利用index 没遇到一个n 则 indx = abs(n)-1 num[index]变成负的 如果 遇到同样的index 则查看的时候会发现 index所处的数在之前被modify 过了 所以 答案append 这个数
```

450. Delete Node in a BST

472. Concatenated Words

```
Dfs 如果一个word的prefix 在words里 就判断是否他的后面这些也在words里 如果在 就是两个单词的情况直接return True 如果不是则递归dfs(后面的)

因为有可能是由3-n个单词组成 只要大于2个 就会return True
```

491. Increasing Subsequences

```
Trick 用字典记录used 过的数字， res.append(path[:]) 因为path 时刻有可能在变化

整体过程 recursion 当path长度大于2 就append

每次记录用过的数字

分支

4-46-467，下一个也是7用过了就continue 结束back to 46 -4677

然后46 这个分支结束了pop 掉6 轮到 47， 47结束了pop掉 7 ，之后7 又用欧了 接着pop掉 所以 4 这个分支结束 pop掉4 接着append 6 
```

494. Target Sum

```
DP 不停更换hashtable ex 记录所有分支的值 和组合成这个值的方法数 依次累加。

n=len(nums) n==1时 值只有-1，1 之后 再来一个1 就会是 -2：1 0：2 2：2 再来一个 -3 -1 1 3 在每个分支下再来一个1 又多出两个分支 但一些分支的值会重叠，此时累加这些相同值的分支，组成新的dic为下次循环使用
```

518. Coin Change 2

```
循环每个coin：在这个循环下再循环coin到amount，这样就不会出现重复。当coin为1的时候，循环1-amount 这样都是1的组合，之后2来的时候只会是2+1的组合不会有1+2 的组合了   
```

523. Continuous Subarray Sum 

```
利用余数来做：s来记录每次循环的后的总和 如果这个总和%k 的余数 在dict 里见过 说明我们加了一个K的倍数到s里 所以肯定是有subarray 之和是k的倍数 return True Note 注意 0 的操作 d[0]=-1
```

536. Construct Binary Tree from String

```
运用stack 遇到 ） 说明 上一个node 的 左右child 都满足了 所以 stack.pop() elif 如果不是）说明只是数字 等下一个不是数字的时候 TreeNode() 这个数字str

然后如果上一个没有left child 那这个就是上一个的left child 反之 是上一个right child 然后 stack.append(这个node) num=""
```

542. 01 Matrix

```
BFS 先把matrix 里面等于1的坐标放到queue里 while queue count=len（queue）用来计算l此时level下节点的个数

如果一个节点的前后左右的level都>=level matrix[x][y]=level+1 queue.append(x,y) than count-=1 when count =0 start a next round which upper level position
```

556. Next Greater Element III

```
先从最后一个往前遍历找到后一个比前一个大的就停止，前一个就是要被替换的那个元素，然后再在最后到找到的那个index 遍历，最先找到的比那个大的就把那两个swap，得到结果，比如1243 先找到的是2 但不能和4换 因为3 也比2大 最后答案是1324 因为最后swap完还要sort一下还完元素之后的所有元素
```

560. Subarray Sum Equals K :

```
每次记录所加和的值 遇到相同的count+=1

没次迭代 都找一下是否cur_sum-k 是否在dic里 在的话直接加他的count

Let's remember count[V], the number of previous prefix sums with value V. If our newest prefix sum has value W, and W-V == K, then we add count[V] to our answer.

This is because at time t, A[0] + A[1] + ... + A[t-1] = W, and there are count[V] indices j with j < t-1 and A[0] + A[1] + ... + A[j] = V. Thus, there are count[V] subarrays A[j+1] + A[j+2] + ... + A[t-1] = K.
```

611. Valid Triangle Number  

```
先排序，把 num 弄成 asceding order 然后从大到小遍历，第一边的index=0 第二个边的index=i-1 这样的话 只要这个满足则 第一个边到第二个边的所有index的边都满足 加到结果里 ，之后 第二边的index-1 变小一下做重复操作，同样 不满足的话说明第一边小了，它的index+1 就可以
```

616. Add Bold Tag in String

```
运用Trie，来查找word，返回查找到的word的最后一个字母index，append到temp res里，通过temp res 得到最后的final

Class Trie():
	def __init__(self):
		self.data={}
	def addword(self,word):
		temp=self.data
		for c in word:
			if c not in temp:
				break
			i+=1
			temp=temp[c]
		for j in word[i:]:
			temp[j]={}
			temp=temp[j]
		temp["#"]={}
	def search(self,index,s):
		ret=-1
		temp=self.data
		for i in range(index,len(s)):
			if s[i] not in temp:
				return ret
			if "#" in temp[s[i]]:
				ret=i
			temp=temp[s[i]]
		return ret
		
```



637. Exclusive Time of Functions :

```
利用stack　， stack里只存sign 是 start 的 time 和 ID: 遍历整个logs，如果是start，就append到stack里 且 stack里的最后一个 也就是前一个start的id所last的时间=现在的时间点-上次start结束的时间点。 如果是end，则pop出来上次start开始的时间点 res[id]+=time-pretime+1  且 前一个start的时间点移到此endpoint的时间点+1
```

652. Find Duplicate Subtrees

```
利用字典，且path=str(root.val)+dfs(root.left)+dfs(root.right) dic[path]+=1
```

673. Number of Longest Increasing Subsequence

```
class Solution:
  def findNumberOfLIS(self, nums: List[int]) -> int:
		 sub=[]
		 num_max=[]
		 dic=collections.defaultdict(list)
		 for num in nums:
				index=bisect.bisect_left(sub,num)
				if len(sub)==index:
						sub.append(num)
				else:
						sub[index]=num
		dic[index].append((sum(lens if max_<num else 0 
		for 				lens,max_ in dic[index-1]) or 1 ,num))#dp
		return sum(i for i,_ in dic[len(sub)-1])
```

694. Number of Distinct Islands

```
判断是否两个小岛一样 [(2,3),(3,3),(2,4),(3,4)]-(2,3)=[(0,0),(1,0),(1,0),(1,1)]
```

721. Accounts Merge

```
Union Find: parent[] 的index 作为每个email的 parent，如果同样的email 出现 则 union 当前的index 和 dic里email之前的parent index
```



739. Daily Temperatures

```
利用stack 倒叙遍历,stack.append(i) ，如果stack[-1]里存的index 所处的值小于我所遍历到的值，那就一直pop 直到找到比我现在值大的，没有的话，就不改变ans的值
```

735. Asteroid Collision

```
利用stack，如果是大于0，就append进来，反之如果是小于0，如果当前stack[-1]也小于0 或者stack为空，就也append进来因为撞不上，但如果stack[-1]>0 且 小于 当前asteroid的绝对值，就pop，直到pop不出来为止。若stack[-1]>0且它的值和当前asteroid的绝对值相等，就pop一次
最终返回stack。
```



763. Partition Labels

```
Sliding window, 为了尽可能把元素放到同一个sequence里，last={c:i for I in enumerate(S)}, 递归每次出现的字母的下标如果是last的那个下标，那就append进入res end-start+1.  Hint: end=max(end,last(c))  为了让那个end值最大 为了让里面的元素出现次数最多
```

767. Reorganize String

```
使用heap 存出现的频率 和字母 先pop出出现频率最大的字母 然后用出现频率其次的 将频率最大的modify i.e aaabb —> ababa 将多出的push 进heap

进行下次循环 注意最后pop出的字母 如果出现的频率是1 则加入到结果中 ，如果大于一 说明没办法modify了 return “ ”　
```

801. Minimum Swaps To Make Sequences Increasing 

```
DP : noswap=[0]*n swap=[0]*n.   For I in range(1,n).   

strictly_increasing=A[i]>A[i-1] and B[i]>B[i-1]

strictly_xincreasing=A[i]>B[i-1] and B[i]>A[i-1]

………
```

881. Boats to Save People 

```
Two Points 因为最多两个人
```

939. Minimum Area Rectangle

```
将问题简化，如果两个点(x1,y1) (x2,y2)，作为rectangle的顶点，然后如果存在(x1,y2),(x2,y1)，则能形成rectangle，然后遍历找到最小的就好了。
```



997. Find the Town Judge

```
设定一个分数 被认识-1分 认识的那个人加一分

当自己的分数是N-1时那肯定就是法官，不是return -1
```

1004. Max Consecutive Ones

```
sliding window
```

1008. Construct Binary Search Tree from Preorder Traversal

```
通过binary search 找到比root 大的node 也是右节点的开始 然后 判断 如果这个开始的节点比root小 说明没有右节点了 如果没有 则root。left是节点之前

root.right 是这个节点之后
```

1011. Capacity To Ship Packages Within D Days

```
binary search : 二分法去找到那个capacity 设立一个函数屈判断capacity可不可以 不可以的话说明小了 可以的话就往小里设置
```



1019. Next Greater Node In Linked List

```
503. Next Greater Element II

都是一个道理，设置cache 里面先设置全0或者全1代表没找到next greater 设置个空 stack 然后遍历 stack里存每次遇到的num的index 相当于对之前的记录

如果遇到一个num 比stack里最后面的index的num大 则在cache里改 用while 直到pop掉stack里之前所有比现在这个num小的index

如果是503 直接循环两次就行
```

1027. Longest Arithmetic Sequence

```
DP:因为要计算第i个元素和之前i-1个元素的diffrence，所以没有O(n)的方法，O(n^2). Tricks：dp数组采用每个是一个字典来记录diff的个数，下一个等一前一个diff+1.
```



1029. Two City Scheduling

````
Greedy algorthm 先添加去A和B差距越大的人，这样很好选择，肯定是去花钱少的那个，然后当i>= len//2 时候，转换选择去另一个城市。
````



1031. Maximum Sum of Two Non-Overlapping Subarrays

```diff
--两种情况：
第一种 L在左 M 在右：循环L右边一直保留M长度的数组maxL=max(maxL,A[i-M]-A[i-M-L])
第二种 L在右 M 在左 同理相反
maxM=max(maxM,A[i-L]-A[i-L-M])
每次循环算res
res=max(res,maxL+A[i]-A[i-M],maxM+A[i]-A[i-L])
```

1114. Print in Order

```
threading.Lock() 设置两个锁一个用于一二线程 一个用于二三线程
demo:
lock=threading.Lock()
lock.acquire() ##mutex设置为0 上锁
lock.release() ##mutex设置为1 释放

```





1152. Analyze User Website Visit Pattern

```
首先用dic--->list 收集每个人查过的关键字，利用sort 和zip 进行对time的排序
然后 利用 Counter去 计算每个人每个 3长度的sequence出现的次数 --->
利用 itertools.combinations(x,3) 
然后利用sum把每个人记录的counter 合并，操作是 sum(每个人counter的list， colllections.Counter()) 语法表示通过什么来sum。
最终找到个数最多的且字母表顺序最小的。利用min min(t,key=lambda x: (-t[x],x))

```



1197. Minimum Knight Moves

```
BFS： trick: 转化题目，将(0,0)--->(x,y) 换成(x,y)-->(0,0) 然后使用abs(x),abs(y)，
因为四个象限问题可以整合到一个象限，因为只要绝对值相同的点去(0,0)的步数是相同的，
这样去的direction也缩减到(-2,-1),(-1,-2)两个方向，每次都要取abs整合到第一象限。 
```





1202. Smallest String With Swaps

```
Union find
```

1242. Web Crawler Multithreaded

```
python 多线程：
from threading import Thread, Lock
with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭／线程中锁的自动获取和释放等。 

try:													with open("１.txt") as file:
    f = open('xxx')							 data = file.read()
except:
    print('fail to open')
    exit(-1)
try:
    do something
except:
    do something
finally:
    f.close()
两边相等
所以 with Lock() 不用解锁了

```

```
threads = [Thread(target=thread_func, args=(web,)) for web in pool]
for thread in threads:
	thread.start()---->开启线程
for thread in threads:
	thread.join()----->等待所有线程完成
可以看到，join方法本身是通过wait方法来实现等待的，这里判断如果线程还在运行中的话，则继续等待，如果指定时间到了，或者线程运行完成了，则代码继续向下执行，调用线程就可以执行后面的逻辑了。
```



1242. Search Suggestions System

```PYTHON
ans=[]
pro=sorted(products)
for i,c in enumerate(searchWord):
	pro=[p for p in pro if i<len(p) and p[i]==c]
	ans.append(pro[:3])
return ans
```

