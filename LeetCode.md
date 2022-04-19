

LeetCode

# 数组

## 二分法

二分法可以解决的变形问题如下

![img](https://static001.geekbang.org/resource/image/42/36/4221d02a2e88e9053085920f13f9ce36.jpg?wh=1142*783)

变体一：查找第一个值等于给定值的元素

比如下面这样一个有序数组，其中，a[5]，a[6]，a[7]的值都等于 8，是重复的数据。我们希望查找第一个等于 8 的数据，也就是下标是 5 的元素。

![img](https://static001.geekbang.org/resource/image/50/f8/503c572dd0f9d734b55f1bd12765c4f8.jpg?wh=1142*284)

```java

public int bsearch(int[] a, int n, int value) {
  int low = 0;
  int high = n - 1;
  while (low <= high) {
    int mid = low + ((high - low) >> 1);
    if (a[mid] >= value) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }

  if (low < n && a[low]==value) return low;
  else return -1;
}
```

换一种写法

```java

public int bsearch(int[] a, int n, int value) {
  int low = 0;
  int high = n - 1;
  while (low <= high) {
    int mid =  low + ((high - low) >> 1);
    if (a[mid] > value) {
      high = mid - 1;
    } else if (a[mid] < value) {
      low = mid + 1;
    } else {
      if ((mid == 0) || (a[mid - 1] != value)) return mid;
      else high = mid - 1;
    }
  }
  return -1;
}
```

   如果我们查找的是任意一个值等于给定值的元素，当 a[mid]等于要查找的值时，a[mid]就是我们要找的元素。但是，如果我们求解的是第一个值等于给定值的元素，当 a[mid]等于要查找的值时，我们就需要确认一下这个 a[mid]是不是第一个值等于给定值的元素。

  如果经过检查之后发现 a[mid]前面的一个元素 a[mid-1]也等于 value，那说明此时的 a[mid]肯定不是我们要查找的第一个值等于给定值的元素。那我们就更新 high=mid-1，因为要找的元素肯定出现在[low, mid-1]之间。

变体二：查找最后一个值等于给定值的元素

```c++
int bsearch(vector<int>&a, int target){
	int left = 0,right = a.size()-1;
	while(left<=right){
		mid = left +((right-left)>>1);

		if(a[mid]>target){
			right = mid - 1;
		}else if(a[mid]<target){
			left = mid+1;
		}else{
			if(mid==n-1 || a[mid+1]!=target)return mid;
			else left = mid +1;
		}

	}
	return -1;
}
```

变体三：查找第一个大于等于给定值的元素

现在我们再来看另外一类变形问题。在有序数组中，查找第一个大于等于给定值的元素。比如，数组中存储的这样一个序列：3，4，6，7，10。如果查找第一个大于等于 5 的元素，那就是 6。

```c++
int bsearch(vector<int>a, int target){
	int left = 0,right = a.size()-1;
	while(left<=right){
		int mid = left +((right-left)>>1);
		if(a[mid]>=target){
			if(mid == 0 || a[mid-1] <target)return mid;
			else right = mid - 1;
		}else{
			left = mid +1;
		}
	}
	return -1;
}
```

变体四：查找最后一个小于等于给定值的元素

现在，我们来看最后一种二分查找的变形问题，查找最后一个小于等于给定值的元素。比如，数组中存储了这样一组数据：3，5，6，8，9，10。最后一个小于等于 7 的元素就是 6。是不是有点类似上面那一种？实际上，实现思路也是一样的。

```java

public int bsearch7(int[] a, int n, int value) {
  int low = 0;
  int high = n - 1;
  while (low <= high) {
    int mid =  low + ((high - low) >> 1);
    if (a[mid] > value) {
      high = mid - 1;
    } else {
      if ((mid == n - 1) || (a[mid + 1] > value)) return mid;
      else low = mid + 1;
    }
  }
  return -1;
}
```

#### [540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

给你一个仅由整数组成的有序数组，其中每个元素都会出现两次，唯有一个数只会出现一次。

请你找出并返回只出现一次的那个数。

你设计的解决方案必须满足 O(log n) 时间复杂度和 O(1) 空间复杂度。

```
输入: nums = [1,1,2,3,3,4,4,8,8]
输出: 2
输入: nums =  [3,3,7,7,10,11,11]
输出: 10
1 <= nums.length <= 105
0 <= nums[i] <= 105
```

本题利用采用二分法。

方法一：整个数组的二分查找。

利用数组的性质可以将其分为两个部分：假设只出现一次的元素位于下标 x，由于其余每个元素都出现两次，因此下标 x 的左边和右边都有偶数个元素，数组的长度是奇数。

由于数组是有序的，因此数组中相同的元素一定相邻。对于下标 x 左边的下标 y，如果nums[y]=nums[y+1]，则 y 一定是偶数；对于下标 x 右边的下标 z，如果 nums[z]=nums[z+1]，则 z 一定是奇数。由于下标 x 是相同元素的开始下标的奇偶性的分界，因此可以使用二分查找的方法寻找下标 x。

![image-20220210214206164](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220210214206164.png)

![image-20220210214219235](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220210214219235.png)

```c++
class Solution {
public:
    int singleNonDuplicate(vector<int>& nums) {
        int left = 0,right= nums.size()-1;
        while(left <right){
            int mid = left + right>>1;
            if(nums[mid] == nums[mid^1]){
                left = mid +1;
            }else{
                right = mid;
            }
        }
        return nums[left];

    }
};
```

#### [5219. 每个小孩最多能分到多少糖果](https://leetcode-cn.com/problems/maximum-candies-allocated-to-k-children/)

给你一个 下标从 0 开始 的整数数组 candies 。数组中的每个元素表示大小为 candies[i] 的一堆糖果。你可以将每堆糖果分成任意数量的 子堆 ，但 无法 再将两堆合并到一起。

另给你一个整数 k 。你需要将这些糖果分配给 k 个小孩，使每个小孩分到 相同 数量的糖果。每个小孩可以拿走 至多一堆 糖果，有些糖果可能会不被分配。

返回每个小孩可以拿走的 最大糖果数目 。

```
输入：candies = [5,8,6], k = 3
输出：5
解释：可以将 candies[1] 分成大小分别为 5 和 3 的两堆，然后把 candies[2] 分成大小分别为 5 和 1 的两堆。现在就有五堆大小分别为 5、5、3、5 和 1 的糖果。可以把 3 堆大小为 5 的糖果分给 3 个小孩。可以证明无法让每个小孩得到超过 5 颗糖果

输入：candies = [2,5], k = 11
输出：0
解释：总共有 11 个小孩，但只有 7 颗糖果，但如果要分配糖果的话，必须保证每个小孩至少能得到 1 颗糖果。因此，最后每个小孩都没有得到糖果，答案是 0 。

1 <= candies.length <= 105
1 <= candies[i] <= 107
1 <= k <= 10 12
```

本题数据量比较大，只能采用nlogn时间复杂度的算法，考虑二分法查找。

1、确定查找的区间

我们需要搜寻的目标是小孩子最多可以拿走多少个糖果；小孩一次最少也得拿一个糖果，最多一次拿走全部。

特殊情况单独讨论：小孩子数量大于糖果数量，直接返回0；

区间为left =0，right= sum/k;

2、二分答案进行查找

利用二分查找缩小查找区间，每次对mid进行check。
mid符合条件，那么我们继续找更大的答案，[left,right]->[mid,right];

mid不符合条件(就不能取到mid)，找更小的区间[left,right]->[left,mid-1]。

3写check函数

本题check就是通过遍历candies，看能不能分出k堆糖果满足k个小孩子。

```c++
class Solution {
public:
    int maximumCandies(vector<int>& candies, long long k) {
         //二分法
         long long sum =0;
         for(auto num:candies)sum+=num;
         if(sum<k)return 0;
         long long left = 1, right = sum/k;
         long ans = 0;
         while(left<right){
             long long mid = (left + right+1)/2;
             if(check(candies,mid,k)){
                 left = mid;
             }else{
                 right = mid-1;
             }
         }
         return int(left);
        
    }
    bool check(vector<int>& candies,long long mid, long long k){
        for(auto num:candies){
            k-=num/mid;
        }
        return k<=0;
    }
};
```

#### [2187. 完成旅途的最少时间](https://leetcode-cn.com/problems/minimum-time-to-complete-trips/)

给你一个数组 time ，其中 time[i] 表示第 i 辆公交车完成 一趟旅途 所需要花费的时间。

每辆公交车可以 连续 完成多趟旅途，也就是说，一辆公交车当前旅途完成后，可以 立马开始 下一趟旅途。每辆公交车 独立 运行，也就是说可以同时有多辆公交车在运行且互不影响。

给你一个整数 totalTrips ，表示所有公交车 总共 需要完成的旅途数目。请你返回完成 至少 totalTrips 趟旅途需要花费的 最少 时间。

```
输入：time = [1,2,3], totalTrips = 5
输出：3
解释：
- 时刻 t = 1 ，每辆公交车完成的旅途数分别为 [1,0,0] 。
  已完成的总旅途数为 1 + 0 + 0 = 1 。
- 时刻 t = 2 ，每辆公交车完成的旅途数分别为 [2,1,0] 。
  已完成的总旅途数为 2 + 1 + 0 = 3 。
- 时刻 t = 3 ，每辆公交车完成的旅途数分别为 [3,1,1] 。
  已完成的总旅途数为 3 + 1 + 1 = 5 。
所以总共完成至少 5 趟旅途的最少时间为 3 。
输入：time = [2], totalTrips = 1
输出：2
解释：
只有一辆公交车，它将在时刻 t = 2 完成第一趟旅途。
所以完成 1 趟旅途的最少时间为 2 。
1 <= time.length <= 10^5
1 <= time[i], totalTrips <= 10^7



```

```c++
class Solution {
public:
    bool check(vector<int>&time, long long mid, int trips){
        long long res = 0;
        for(auto &t:time){
            res+= mid/t;
        }
        return res>=trips;
    }
    long long minimumTime(vector<int>& time, int totalTrips) {
        long long left = 1,right = (long long )totalTrips *(*min_element(time.begin(),time.end()));
        long long res;
        while(left<right){
            long long mid = left + (right- left)/2;
            if(check(time,mid,totalTrips)){
                right = mid;
            }else{
                left = mid + 1;
            }
        }
        return left;
        
    }
};
```



## 双指针

#### [26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

```
输入：nums = [1,1,2]
输出：2, nums = [1,2]
输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
0 <= nums.length <= 3 * 104
-104 <= nums[i] <= 104
nums 已按升序排列
```

本题直接采用双指针,当nums[fast]!=nums[fast-1]移动slow指针，将fast指针赋值给slow。

```c++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int slowIndex = 1;
        if(nums.size()==0)return 0;
        for(int fastIndex = 1; fastIndex<nums.size();fastIndex++){
            if(nums[fastIndex] != nums[fastIndex-1]){
                nums[slowIndex++] = nums[fastIndex];
            }
        }
        return slowIndex;


    }
};
```

#### [581. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

给你一个整数数组 `nums` ，你需要找出一个 **连续子数组** ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

请你找出符合题意的 **最短** 子数组，并输出它的长度。

 

**示例 1：**

```
输入：nums = [2,6,4,8,10,9,15]
输出：5
解释：你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
```

**示例 2：**

```
输入：nums = [1,2,3,4]
输出：0
```

**示例 3：**

```
输入：nums = [1]
输出：0
```

 

**提示：**

- `1 <= nums.length <= 104`
- `-105 <= nums[i] <= 105`

 ![微信截图_20200921203355.png](https://pic.leetcode-cn.com/1600691648-ZCYlql-%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200921203355.png)

借助上面的图来进行分析 从左向右遍历，维护一个max，其中right就是最后一个max。当遇到当前数字小于max才会更新right；

从右向左遍历，维护一个min，得到的left就是最后一个min。当遇到当前数字大于min才会更新left；

```c++
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int left = 0,right = -1;
        int max = INT_MIN,min = INT_MAX;
        int len = nums.size();
        for(int i = 0; i<len; i++){
            //从左到右找右边界right
            if(nums[i] < max){
                right = i;
            }else max = nums[i];

            if(nums[len-i-1] > min)left = len-i-1;
            else min = nums[len-i-1];
        }
        return right-left+1;
    }
};
```



# 链表

链表是通过指针串联在一起的线性结构，在此针对单链表。每个链表包含两个部分，一个是数据域还有一个是指针域（后驱节点）。最后一个节点，指向NULL(空指针)。



主要有三种类型的链表。

## 单链表

![链表1](https://img-blog.csdnimg.cn/20200806194529815.png)

##  双链表

双链表存在两个指针域，分别存储前驱节点和后驱节点，可以双向查询。

![链表2](https://img-blog.csdnimg.cn/20200806194559317.png)

## 循环链表

![链表4](https://img-blog.csdnimg.cn/20200806194629603.png)

循环链表要求首尾相连。

和数组不同的是，链表不需要存储在连续空间下，在内存中是不连续分布的。

![链表3](https://img-blog.csdnimg.cn/20200806194613920.png)

```c++
struct ListNode {
    int val;  // 节点上存储的元素
    ListNode *next;  // 指向下一个节点的指针
    ListNode(int x) : val(x), next(NULL) {}  // 节点的构造函数
};
```

包含的一些基本操作如删除、增加、

删除节点

![链表-删除节点](https://img-blog.csdnimg.cn/20200806195114541.png)

添加节点

![链表-添加节点](https://img-blog.csdnimg.cn/20200806195134331.png)

![链表-链表与数据性能对比](https://img-blog.csdnimg.cn/20200806195200276.png)

#### [203. 移除链表元素](https://leetcode-cn.com/problems/remove-linked-list-elements/)

给你一个链表的头节点 `head` 和一个整数 `val` ，请你删除链表中所有满足 `Node.val == val` 的节点，并返回 **新的头节点** 。

![img](https://assets.leetcode.com/uploads/2021/03/06/removelinked-list.jpg)

```
输入：head = [1,2,6,3,4,5,6], val = 6
输出：[1,2,3,4,5]
输入：head = [], val = 1
输出：[]
输入：head = [7,7,7,7], val = 7
输出：[]
列表中的节点数目在范围 [0, 104] 内
1 <= Node.val <= 50
0 <= val <= 50
```

本题利用哨兵节点 不需要分情况讨论。

```c++
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if(head == NULL)return head;
        ListNode *preHead = new ListNode(0);
        preHead->next = head;
        ListNode * cur = preHead;
        while(cur->next){
            if(cur->next->val == val){
                ListNode *node = cur->next;
                cur->next = cur->next->next;
                delete node;

            }
            else cur = cur->next;
        }
        return preHead->next;
    }
};
```



#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)

```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex2.jpg)

```
输入：head = [1,2]
输出：[2,1]
输入：head = []
输出：[]
链表中节点的数目范围是 [0, 5000]
-5000 <= Node.val <= 5000
```

本题链表反转，利用双指针将当前节点的next指针改为前一个节点。

pre指针存储前一个节点的引用，cur保存当前节点。

```c++
ListNode* reverseList(ListNode* head) {
        ListNode *cur = head;
        ListNode *pre = NULL;
        ListNode *temp;
        while(cur){
            temp = cur->next;
            cur->next = pre;
            //更新操作
            pre = cur;
            cur = temp;
            
         }
        return pre；

    }
```

当然也可以利用递归，但是递归的做法更加抽象，不太容易想象，最好在画出来。

假设链表为：
$$
n 
1
​
 →…→n 
k−1
​
 →n 
k
​
 →n 
k+1
​
 →…→n 
m
​
 →∅
 
$$
若从节点 n**k*+1 到 n**m* 已经被反转，而我们正处于 n**k*。
$$
n 
1
​
 →…→n 
k−1
​
 →n 
k
​
 →n 
k+1
​
 ←…←n 
m
​
$$
希望nk+1指向nk;

```c++
ListNode* reverse(ListNode *pre,ListNode *cur){
        if(cur==NULL)return pre;
        ListNode* temp = cur->next;
        cur->next = pre;
        return reverse(cur,temp);
    }
ListNode* reverseList(ListNode* head) {
    return reverse(NULL,head);

}
```

#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

![img](https://assets.leetcode.com/uploads/2020/10/03/swap_ex1.jpg)

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]
输入：head = []
输出：[]
输入：head = [1]
输出：[1]
链表中节点的数目在范围 [0, 100] 内
0 <= Node.val <= 100
```

这道题最好还是先设立虚节点。方便处理头结点交换。

cur为当前节点，当cur后面没有节点或者只有一个节点，那么就交换结束。

否则，获得cur后面两个节点temp1,temp2，通过更新节点的指针关系实现两两交换节点。

![img](https://assets.leetcode-cn.com/solution-static/24/6.png)

```c++

class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *preHead = new ListNode(0);//虚拟头结点
        preHead->next = head;
        ListNode *cur = preHead;
        if(head == NULL|| head->next==NULL)return head;
        while(cur->next!=nullptr && cur->next->next!=nullptr)   {
            ListNode * temp1 = cur->next;
            ListNode *temp2 = cur->next->next;

            cur->next = temp2;
            temp1->next = temp2->next;
            temp2->next = temp1;
            cur = temp1;

        }
        return preHead->next;

    }
};
```

#### [剑指 Offer II 026. 重排链表](https://leetcode-cn.com/problems/LGjMqU/)

给定一个单链表 L 的头节点 head ，单链表 L 表示为：

 L0 → L1 → … → Ln-1 → Ln 
请将其重新排列后变为：

L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …

不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

![img](https://pic.leetcode-cn.com/1626420311-PkUiGI-image.png)

```
输入: head = [1,2,3,4]
输出: [1,4,2,3]
```

![img](https://pic.leetcode-cn.com/1626420320-YUiulT-image.png)

```
输入: head = [1,2,3,4,5]
输出: [1,5,2,4,3]
链表的长度范围为 [1, 5 * 104]
1 <= node.val <= 1000
```

方法1 利用线性表随机访问的性质，利用数组存储链表，直接顺序访问元素即可重建链表。

```c++

class Solution {
public:
    void reorderList(ListNode* head) {
        //将链表存储到线性表中
        if(head == nullptr)return;

        vector<ListNode*> vec;
        ListNode *cur = head;
        while(cur!= NULL){
            vec.push_back(cur);
            cur = cur->next;
        }
        //双指针
        int i = 0, j = vec.size()-1;
        while(i<j){
            vec[i]->next = vec[j];
            i++;
            if(i == j)break;;
            vec[j]->next = vec[i];
            j--;
        }
        vec[i]->next = nullptr;
        


    }
};
```

方法2 链表逆序+找链表中点+ 合并

```c++
class Solution {
public:
    void reorderList(ListNode* head) {
        if(head == nullptr)return;

        ListNode *mid = middleNode(head);
        ListNode* l1 = head;
        ListNode *l2 = mid->next;
        mid->next = nullptr;
        l2 = reverseList(l2);
        mergeList(l1,l2);


    }
    ListNode* middleNode(ListNode *head){
        ListNode *slow = head;
        ListNode *fast = head;
        while(fast->next != nullptr && fast->next->next!= nullptr){
            fast =fast->next->next;
            slow = slow->next;
        }
        return slow;
    }
    ListNode *reverseList(ListNode *head){
        ListNode *cur = head;
        ListNode *pre = NULL;
        while(cur){
            ListNode *tmp = cur->next;
            cur->next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    }
    void mergeList(ListNode *l1,ListNode *l2){
        ListNode *temp1;
        ListNode *temp2;

        while(l1!=nullptr && l2!=nullptr){
            temp1 = l1->next;
            temp2 = l2->next;

            l1->next = l2;
            l1 = temp1;

            l2->next = l1;
            l2 = temp2;
        }
    }
};
```



#### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。 

为了让您更好地理解问题，以下面的二叉搜索树为例：

![img](https://assets.leetcode.com/uploads/2018/10/12/bstdlloriginalbst.png)

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

![img](https://assets.leetcode.com/uploads/2018/10/12/bstdllreturndll.png)

利用中序遍历取出二叉搜索数元素。并构建双向链表，其中pre->right = cur, cur->left = pre；

最后利用head初始化循环链表head->left = pre,pre->right = head; 

```c++
class Solution {
public:
    Node *pre, *head;
    //对二叉搜索树进行中序遍历 pre 和 cur
    void dfs(Node *cur){
        if(cur == nullptr)return;
        dfs(cur->left);
        //处理为双向链表
        //pre 用于记录当前节点左侧的节点 即上一个节点
        if(pre!=nullptr)pre->right = cur;
        //pre为空时表示的是 当前为头结点
        else head = cur;
        cur->left = pre;
        //更新操作
        pre = cur;
        dfs(cur->right);
    }
    Node* treeToDoublyList(Node* root) {
        if(root == nullptr)return nullptr;
        dfs(root);
        //初始化为循环链表
        pre->right = head;
        head->left = pre;//进行首尾节点的相互指向

       
        return head;
        
    }
};
```



# 栈和队列

### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。

2. 左括号必须以正确的顺序闭合。

   ```
   输入：s = "()"
   输出：true
   输入：s = "()[]{}"
   输出：true
   输入：s = "(]"
   输出：false
   输入：s = "([)]"
   输出：false
   ```

括号匹配符合栈的运算。

编译原理的话，编译器在 词法分析的过程中处理括号、花括号等这个符号的逻辑，也是使用了栈这种数据结构。

不匹配有三种情况

1. 第一种情况，字符串里左方向的括号多余了 ，所以不匹配。 ![括号匹配1](https://img-blog.csdnimg.cn/2020080915505387.png)

2. 第二种情况，括号没有多余，但是 括号的类型没有匹配上。 ![括号匹配2](https://img-blog.csdnimg.cn/20200809155107397.png)

3. 第三种情况，字符串里右方向的括号多余了，所以不匹配。 ![括号匹配3](https://img-blog.csdnimg.cn/20200809155115779.png)

   ```c++
   class Solution {
   public:
       bool isValid(string s) {
           stack<char>st;
           for(int i= 0; i<s.size(); i++){
               if(s[i] == '(')st.push(')');
               else if(s[i] == '[')st.push(']');
               else if(s[i] =='{')st.push('}');
               else if(st.empty() || st.top()!=s[i])return false;
               else  st.pop();
           }
   
           return st.empty();
   
       }
   };
   ```

   方法2：利用哈希表建立对应映射关系

   当我们遇到一个左括号时，我们会期望在后续的遍历中，有一个相同类型的右括号将其闭合。由于**后遇到的左括号要先闭合**，因此我们可以将这个左括号放入栈顶。

   当我们遇到一个右括号时，我们需要将一个相同类型的左括号闭合。此时，我们可以取出栈顶的左括号并判断它们是否是相同类型的括号。如果不是相同的类型，或者栈中并没有左括号，那么字符串 ss 无效，返回 \text{False}False。为了快速判断括号的类型，我们可以使用哈希表存储每一种括号。哈希表的键为右括号，值为相同类型的左括号。

   ```c++
   bool isValid(string s) {     
   		int n = s.size();
           if(n%2 == 1)return false;
           unordered_map<char,char>pairs={
               {')','('},
               {']','['},
               {'}','{'},
           };
           stack<char>st;
           for(auto ch:s){
               //count()是返回按 键 搜索的个数，只有右括号是键，
               //所以当字符是右括号时返回1，左括号返回0. 右括号去栈顶查找，左括号直接入栈
               if(pairs.count(ch)){
                   if(st.empty() || st.top()!= pairs[ch])return false;
                   st.pop();
               }else{
                   st.push(ch);
               }
           }
   
           return st.empty();
   
       }
   ```

### [150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

根据[ 逆波兰表示法](https://baike.baidu.com/item/逆波兰式/128437)，求表达式的值。

有效的算符包括 `+`、`-`、`*`、`/` 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

- 整数除法只保留整数部分。
- 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

```
输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
逆波兰表达式：

逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。

平常使用的算式则是一种中缀表达式，如 ( 1 + 2 ) * ( 3 + 4 ) 。
该算式的逆波兰表达式写法为 ( ( 1 2 + ) ( 3 4 + ) * ) 。
逆波兰表达式主要有以下两个优点：
去掉括号后表达式无歧义，上式即便写成 1 2 + 3 4 + * 也可以依据次序计算出正确结果。适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中。

```

逆波兰表达式遵循从左往右运算，计算时用stack存储操作数，从左往右遍历整个表达式：

如果遇到操作数，则将操作数入栈；

如果遇到运算符，则将两个操作数出栈，其中先出栈的是右操作数，后出栈的是左操作数，使用运算符对两个操作数进行运算，将运算得到的新操作数入栈。

```c++
class Solution {
public:
    bool isNumber(string &token){
        return !(token=="+"||token=="-"||token=="*"||token=="/");
    }
    int evalRPN(vector<string>& tokens) {
        stack<int>st;
        for(int i = 0; i<tokens.size(); i++){
            if(!isNumber(tokens[i])){
                int num1 = st.top();
                st.pop();
                int num2 = st.top();
                st.pop();
                if(tokens[i]=="+")st.push(num1+num2);
                else if(tokens[i]=="-")st.push(num2-num1);
                else if(tokens[i]=="*")st.push(num2*num1);
                else st.push(num2/num1);

            }else{
                st.push(stoi(tokens[i]));
            }
            

        }
        return st.top();

    }
};
```

### [1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

```
输入："abbaca"
输出："ca"
```

和有效括号题类似，利用栈的性质。

消除一对相邻重复项可能会导致新的相邻重复项出现，如从字符串 abba 中删除 bb 会导致出现新的相邻重复项aa 出现。

```c++
 //c++中字符串string 有入栈出栈的接口
class Solution {
public:
    string removeDuplicates(string s) {
		string res;
        for(char ch:s){
            if(res.empty() || res.back()!=ch)res.push_back(ch);
            else res.pop_back();
        }
        return res;
    }
};
```

# 二叉树

#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

翻转一棵二叉树。

```
     4
   /   \
  2     7
 / \   / \
1   3 6   9

     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

方法1 深度优先遍历 翻转二叉树，其实对应翻转每一个左右子树就可以实现。 

![226.翻转二叉树1](https://img-blog.csdnimg.cn/20210203192724351.png)

那么前序遍历和后序遍历都是比较容易解决的。

后序遍历递归法实现

```c++
class Solution {
public:

    TreeNode* invertTree(TreeNode* root) {
        if(root == NULL)return root;
        swap(root->left,root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```

迭代法实现：迭代我就直接采用的前序遍历，后序遍历的话还需要进行一次逆序。

```c++
 TreeNode* invertTree(TreeNode* root) {
        if (root == NULL)return root;
        stack<TreeNode*>st;
        st.push(root);
        while(!st.empty()){
            TreeNode *node = st.top();
            st.pop();
            swap(node->left,node->right);
            if(node->right)st.push(node->right);
            if(node->left)st.push(node->left);

        }
        return root;
    }
```



#### [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。

假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

![img](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)



```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Input: preorder = [-1], inorder = [-1]
Output: [-1]
0 <= 节点个数 <= 5000
```

题解思路：分治算法

前序遍历性质： 节点按照 [ 根节点 | 左子树 | 右子树 ] 排序。
中序遍历性质： 节点按照 [ 左子树 | 根节点 | 右子树 ] 排序。

以题目示例为例：

前序遍历划分 [ 3 | 9 | 20 15 7 ]
中序遍历划分 [ 9 | 3 | 15 20 7 ]

1.通过遍历前序遍历的首元素,为 根节点node的值。

2、在中序遍历中找到根节点node 的下标索引，将其分为左 根 右

3、在中序遍历中得到左右子树的区间（元素个数），可以将前序遍历划分为根 左 右。

![Picture1.png](https://pic.leetcode-cn.com/1629825510-roByLr-Picture1.png)

利用分治算法的思想，可以不断分成更小的子节点。可以求得1、根节点

2、左子树3、右子树

递归步骤：1、参数：根节点在前序遍历的索引 子树的中序遍历左边界 右边界。

2、停止条件 ：left>right 

3、递归

先建立根节点 preorder[root]

划分左右子树

构建左右子树 （左右递归）

	根节点索引	中序遍历左边界	中序遍历右边界
左子树	root + 1	left	i - 1
右子树	i - left + root + 1	i + 1	right

```c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        this->preorder = preorder;
        for(int i = 0; i<preorder.size(); i++){
            dic[inorder[i]] = i;
        }
        return recur(0,0,inorder.size()-1);
    }
    private:
    vector<int>preorder;
    //利用哈希表建立映射关系 就不用遍历找节点了
    unordered_map<int,int>dic;
    TreeNode *recur(int root,int left, int right){
        if(left>right)return nullptr;
        TreeNode *node = new TreeNode(preorder[root]);
        //利用先序遍历的根节点在中序中找对应 下标
        int i = dic[preorder[root]];
        //递归参数分别为：
        //(左右子树)先序遍历根节点下标 中序遍历的左子树区间 右子树区间
        node->left = recur(root+1,left,i-1);
        //先序遍历右子树的根节点下标
        //也就是当前根节点加左子树长度root+(i-1-left+1)最后加1
        node->right = recur(root+i-left+1,i+1,right);
        return node;
    }
};
```

#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

根据一棵树的中序遍历与后序遍历构造二叉树。

**注意:**
你可以假设树中没有重复的元素。

```
中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
```

```
  3
   / \
  9  20
    /  \
   15   7
```

题解思路：

本题和上一道题基本一样的。

题解思路：分治算法

前序遍历性质： 节点按照 [ 根节点 | 左子树 | 右子树 ] 排序。
中序遍历性质： 节点按照 [ 左子树 | 根节点 | 右子树 ] 排序。

以题目示例为例：

后序遍历划分 [ 9 | 15  7 20 |3]
中序遍历划分 [ 9 | 3 | 15 20 7 ]

1.通过遍历后序遍历的尾元素,为 根节点node的值。

2、在中序遍历中找到根节点node 的下标索引，将其分为左 根 右

3、在中序遍历中得到左右子树的区间（元素个数），可以将后序遍历划分为 左 右 根。

```c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        this->postorder = postorder;
        for(int i=0; i<inorder.size();i++){
            dic[inorder[i]] = i;
        }
        return recur(postorder.size()-1,0,postorder.size()-1);


    }
private:
    vector<int>postorder;
    unordered_map<int,int>dic;
    TreeNode *recur(int root,int left,int right){
        if(left>right)return nullptr;
        TreeNode *node = new TreeNode(postorder[root]);
        int i = dic[postorder[root]];
        //主要改动只有 （左右子树）后序遍历对应根节点的位置
        //右子树在前序遍历的根节点下标 当前根节点下标-1
        //左子树 当前根节点下标减去右子树的长度 root-(right-i-1+1)-1
        node->right = recur(root-1,i+1,right);
        node->left = recur(root-1-right+i, left, i-1);
        
        return node;

    }
};
```

#### [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/15/117_sample.png)

```
输入：root = [1,2,3,4,5,null,7]
输出：[1,#,2,3,#,4,5,7,#]
树中的节点数小于 6000
-100 <= node.val <= 100
```

本题采用二叉树的层序遍历，对于每一层我们将节点放入队列里，记录当前队列的大小。

在每层队列内，我们把直到最后一个元素之前的所有元素，构成链表，每层的最后一个节点指向NULL.



```c++
class Solution {
public:
    Node* connect(Node* root) {
        queue<Node*>que;
        if(root!=NULL)que.push(root);
        while(!que.empty()){
            int size = que.size();
            Node* node;
            for(int i = 0; i<size; i++){
                node = que.front();
                que.pop();
                if(i<size-1){
                    node->next = que.front();
                    
                }

                if(node->left)que.push(node->left);
                if(node->right)que.push(node->right);

            }
            node->next == nullptr;
        }
        return root;
        
    }
};
```

#### [116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

给定一个 **完美二叉树** ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

```
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

![img](https://assets.leetcode.com/uploads/2019/02/14/116_sample.png)

```
输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
```

本题除了是完美二叉树外和上一题没有区别。

```c++
class Solution {
public:
    Node* connect(Node* root) {
        queue<Node*>que;
       
        if(root!=NULL)que.push(root);
        while(!que.empty()){
            int size = que.size();
            Node* node;
            for(int i =0; i<size; i++){
                node = que.front();
                que.pop();
                if(i<size-1){
                    node->next = que.front();
                }

                if(node->left)que.push(node->left);
                if(node->right)que.push(node->right);

            }
            node->next = NULL;
        }
        return root;
        
    }
};
```

#### [654. 最大二叉树](https://leetcode-cn.com/problems/maximum-binary-tree/)

给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：

二叉树的根是数组 nums 中的最大元素。
左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
返回有给定数组 nums 构建的 最大二叉树 。

![img](https://assets.leetcode.com/uploads/2020/12/24/tree1.jpg)

```

输入：nums = [3,2,1,6,0,5]
输出：[6,3,5,null,2,0,null,null,1]
```

![img](https://assets.leetcode.com/uploads/2020/12/24/tree2.jpg)

```
输入：nums = [3,2,1]
输出：[3,null,2,null,1]
1 <= nums.length <= 1000
0 <= nums[i] <= 1000
nums 中的所有整数 互不相同
```

构造树一般采用的是前序遍历，因为先构造中间节点，然后递归构造左子树和右子树。

- 确定递归函数的参数和返回值

  参数就是传入的是存放元素的数组，返回该数组构造的二叉树的头结点，返回类型是指向节点的指针。

  ```c++
  TreeNode* constructMaximumBinaryTree(vector<int>& nums)
  ```

  确定终止条件

  ```c++
  TreeNode* node = new TreeNode(0);
  if (nums.size() == 1) {
      node->val = nums[0];
      return node;
  }
  ```

  - 确定单层递归的逻辑
  - 1、先要找到数组中最大的值和对应的下标， 最大的值构造根节点，下标用来下一步分割数组。
  - 2最大值所在的下标左区间 构造左子树
  - 3最大值所在的下标右区间 构造右子树

```c++
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        TreeNode *node = new TreeNode(0);
        if(nums.size() == 1){
            node->val = nums[0];
            return node;
        }
        //确定递归逻辑 1找每次递归时最大的数和对应下标 利用最大的数构造根节点 
        //下标用来分割数组

        int maxValue = 0;
        int maxValueIndex = 0;
        for(int i = 0; i<nums.size(); i++){
            if(nums[i]>maxValue){
                maxValue = nums[i];
                maxValueIndex = i;
            }
        }
        node->val = maxValue;

        //最大值的左侧 构造左子树
        if(maxValueIndex>0){
            vector<int> newVec(nums.begin(), nums.begin()+maxValueIndex);
            node->left = constructMaximumBinaryTree(newVec);
        }

        //最大值右侧 右子树
        if(maxValueIndex<nums.size()-1){
            vector<int>newVec(nums.begin()+maxValueIndex+1,nums.end());
            node->right = constructMaximumBinaryTree(newVec);
        }
        return node;


    }
};
```

1. 

#### [501. 二叉搜索树中的众数](https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/)

给你一个含重复值的二叉搜索树（BST）的根节点 root ，找出并返回 BST 中的所有 众数（即，出现频率最高的元素）。

如果树中有不止一个众数，可以按 任意顺序 返回。

假定 BST 满足如下定义：

结点左子树中所含节点的值 小于等于 当前节点的值
结点右子树中所含节点的值 大于等于 当前节点的值
左子树和右子树都是二叉搜索树

![img](https://assets.leetcode.com/uploads/2021/03/11/mode-tree.jpg)

```
输入：root = [1,null,2,2]
输出：[2]
```

```
输入：root = [0]
输出：[0]
```

二叉搜索树天然具有顺序，所以求众数只需要顺序遍历统计即可，对于普通二叉树，则需要利用map统计每一个出现的次数，再排序求解。、

方法1 递归法的深度优先遍历

利用pre指针保存上一个遍历节点的值，就不用再单独开数组来存储所有值了。

记得更新 pre = cur;



```c++
class Solution {
public:
    //二叉搜索树的特性 本来就是有序的
    //所以和正常二叉树相比 不需要在用map统计次数排序了
    TreeNode *pre = NULL;
    vector<int>res;
    int count = 0,maxCount = INT_MIN;
    void traversal(TreeNode *node){
        
        if(node == NULL)return;
        traversal(node->left);
        if(pre == NULL){
            count = 1;
        }else if(pre->val == node->val){
            count++;
        }else {
            count = 1;
        }
        pre = node;
        if(count == maxCount){
            res.push_back(node->val);
        }
        if(count > maxCount){
            maxCount = count;
            //那么之前加入的res就不是众数了
            res.clear();
            res.push_back(node->val);

        }
        traversal(node->right);

    }
    vector<int> findMode(TreeNode* root) {
        
        traversal(root);
        return res;

    }
};
```



#### [1036. 逃离大迷宫](https://leetcode-cn.com/problems/escape-a-large-maze/)

在一个 106 x 106 的网格中，每个网格上方格的坐标为 (x, y) 。

现在从源方格 source = [sx, sy] 开始出发，意图赶往目标方格 target = [tx, ty] 。数组 blocked 是封锁的方格列表，其中每个 blocked[i] = [xi, yi] 表示坐标为 (xi, yi) 的方格是禁止通行的。

每次移动，都可以走到网格中在四个方向上相邻的方格，只要该方格 不 在给出的封锁列表 blocked 上。同时，不允许走出网格。

只有在可以通过一系列的移动从源方格 source 到达目标方格 target 时才返回 true。否则，返回 false。

```
输入：blocked = [[0,1],[1,0]], source = [0,0], target = [0,2]
输出：false
解释：
从源方格无法到达目标方格，因为我们无法在网格中移动。
无法向北或者向东移动是因为方格禁止通行。
无法向南或者向西移动是因为不能走出网格。
输入：blocked = [], source = [0,0], target = [999999,999999]
输出：true

0 <= blocked.length <= 200
blocked[i].length == 2
0 <= xi, yi < 106
source.length == target.length == 2
0 <= sx, sy, tx, ty < 106
source != target
题目数据保证 source 和 target 不在封锁列表内


```

本题如果直接对整个棋盘进行搜索的话，肯定会超时，因为整个棋盘的数据是1e6*1e6.

那么我们考虑是不是可以缩小范围，从反面来思考。

什么时候source 和 target不能连通，也就是source或者target被障碍物包围起来的时候。一个很容易想到的思路是：从 s 跑一遍 BFS，然后从 t跑一遍 BFS，同时设定一个最大访问点数量 MAX，**若从两者出发能够访问的点数量都能超过 MAX，说明两点均没有被围住，最终必然会联通。**

那么max最大范围怎么去界定呢，肯定和blocked数组相关。**在给定数量障碍物的前提下，障碍物所能围成的最大面积为多少。**

首先，容易想到：**任何一条封闭图形的直边都可以通过调整为斜边来围成更大的面积：**

![image.png](https://pic.leetcode-cn.com/1641855571-IOaJZJ-image.png)

当然如果要达到最大面积，那么必须要利用棋盘边界，来形成封闭图形。

![image.png](https://pic.leetcode-cn.com/1641856898-BYFygs-image.png)

也就是上面的形状。根据「等差数列求和」可知，如果从 s和 t出发，能够访问的点数超过n∗(n−1) 个。

```c++
class Solution {
public:
    int qisize = 1e6,max = 5e4;
    long long BASE = 13331;
    unordered_set<long long>set;
     int dir[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
    bool isEscapePossible(vector<vector<int>>& blocked, vector<int>& source, vector<int>& target) {
        //障碍物少 不能形成包围圈
        //哈希映射  将二维block坐标映射为一位数字
       for(auto &p :blocked)set.insert(p[0]*BASE+p[1]);
       if(blocked.size()<2)return true;
       int n = blocked.size();
       max = n*(n-1)/2;
       return check(source,target) &&check(target,source);


    }

    //检查source target 是否被包围了
    bool check(vector<int>&a, vector<int>&b){
       unordered_set<long long>vis;
       //广度优先
       queue<pair<int,int>>que;
       que.push({a[0],a[1]});
       vis.insert(a[0]*BASE+a[1]);
       while(!que.empty() && vis.size()<=max){
           auto t = que.front();
           que.pop();
           int x = t.first, y = t.second;
           if(x == b[0]&& y == b[1])return true;
           for(int i =0; i<4; i++){
               int nx = x+dir[i][0], ny = y+dir[i][1];
               if(nx<0 || nx>=qisize || ny<0 || ny>=qisize)continue;
               if(vis.count(nx*BASE+ny))continue;
               if(set.count(nx*BASE+ny))continue;
               que.push({nx,ny});
               vis.insert(nx*BASE+ny);
           }

       }

       return vis.size()>max;

    }
};
```



# 位运算

#### [剑指 Offer II 001. 整数除法](https://leetcode-cn.com/problems/xoh6Oh/)

给定两个整数 `a` 和 `b` ，求它们的除法的商 `a/b` ，要求不得使用乘号 `'*'`、除号 `'/'` 以及求余符号 `'%'` 。

整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231, 231−1]。本题中，如果除法结果溢出，则返回 231 − 1

```
输入：a = 15, b = 2
输出：7
解释：15/2 = truncate(7.5) = 7
输入：a = 7, b = -3
输出：-2
解释：7/-3 = truncate(-2.33333..) = -2
输入：a = 0, b = 1
输出：0
输入：a = 1, b = 1
输出：1
-231 <= a, b <= 231 - 1
b != 0
```

最初想法：不能使用、/ * %等符号，那么只能采用减法，一种方式就是不停的减，

第一种思路逐次减法：会超时，代码如下：

需要考虑的是边界问题，因为负数最多是-2^31 正数最多是2^31-1;

所以就把被除数和除数都转化为负数。

```c++
int divide(int a, int b) {
    //对于边界情况单独考虑 int范围-2147483648 到 2147483647
    if (a == INT_MIN && b == -1) return INT_MAX;

    int sign = (a > 0) ^ (b > 0) ? -1 : 1;

    if (a > 0) a = -a;
    if (b > 0) b = -b;
    
    unsigned int res = 0;
    while (a <= b) {
        a -= b;
        res++;
    }

    // bug 修复：因为不能使用乘号，所以将乘号换成三目运算符
    return sign == 1 ? res : -res;
}


```

对思路1 进行优化。比如20/3，用20不停的减去3，但这种效率太差。可以用20减去3的倍数，我们发现20减去6要比减去3更快，当不够减的时候再用他减去3。我们还发现用20减去12的时候要比减去6更快，所以发现一个规律，就是把除数不停的往左移，当除数离被除数最近的时候就用被除数减去除数。

```c++
class Solution {
public:
    int divide(int a, int b) {
        if(a== INT_MIN && b == -1)return INT_MAX;

        //a b如果同号商为正 否则商为负数
        int sign = (a>0) ^(b>0)? -1:1;
        unsigned int ua = abs(a);
        unsigned int ub = abs(b);

        unsigned int res = 0;
        for(int i = 31; i>=0; i--){
            if((ua>>i) >= ub){
                ua = ua - (ub<<i);
                res+= 1<<i;

            }
        }

        return sign == 1?res:-res;

    }
};
```



#### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

```
输入: a = 1, b = 1
输出: 2
a, b 均可能是负数或 0
结果不会溢出 32 位整数
```

![image-20220107104218193](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220107104218193.png)

无进位的和与异或相同，进位部分和与运算相同，但是结果需要左移一位。

![image-20220107104321977](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220107104321977.png)



```c++
class Solution {
public:
    int add(int a, int b) {
        //c++中负数不支持左移操作
        while(b!=0){
            //保存进位值，下次循环用
            int c =(unsigned int) (a&b)<<1;
            //保存不进位值，下次循环用，
            a ^=b;
            //如果还有进位，再循环，如果没有，则直接输出没有进位部分即可。
            b = c;

        }
        return a;

    }
};
```

#### [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。

 

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用 二进制补码 记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。

lowbit(x)是x的二进制表达式中最低位的1所对应的值。

int lowbit(int x)
{
    return x&(-x);
}



```c++
class Solution {
public:
    int lowbit(uint32_t x){
        return x&-x;
    }
    int hammingWeight(uint32_t n) {
        int res = 0;
        while(n){
            
           n -= lowbit(n);
           res++;
        }

        return res;
        
    }
};
```

#### [剑指 Offer II 004. 只出现一次的数字 ](https://leetcode-cn.com/problems/WGki4K/)

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次 。**请你找出并返回那个只出现了一次的元素。

```
输入：nums = [2,2,3,2]
输出：3
输入：nums = [0,1,0,1,0,1,100]
输出：100
1 <= nums.length <= 3 * 104
-231 <= nums[i] <= 231 - 1
nums 中，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次


```

方法1 利用哈希表统计次数，找到只出现一次的数字

方法一比较容易想到，但是空间复杂度是O(n)的。

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        unordered_map<int,int>hash(nums.size());
        int res = 0;
        for(int i = 0; i<nums.size(); i++){
            hash[nums[i]] ++;
        }
        for(int i = 0; i<nums.size(); i++){
            if(hash[nums[i]] == 1)res = nums[i];
        }
        return res;
    }
};
```

方法2 有限状态自动机

下图所示，考虑数字的二进制形式，对于出现三次的数字，各 二进制位 出现的次数都是 3 的倍数。
因此，统计所有数字的各二进制位中 1 的出现次数，并对 3 求余，结果则为只出现一次的数字。

各二进制位的 位运算规则相同 ，因此只需考虑一位即可。如下图所示，对于所有数字中的某二进制位 1 的个数，存在 3 种状态，即对 3 余数为 0,1,2 。

![Picture1.png](https://pic.leetcode-cn.com/28f2379be5beccb877c8f1586d8673a256594e0fc45422b03773b8d4c8418825-Picture1.png)

![Picture2.png](https://pic.leetcode-cn.com/ab00d4d1ad961a3cd4fc1840e34866992571162096000325e7ce10ff75fda770-Picture2.png)

![Picture3.png](https://pic.leetcode-cn.com/0a7ea5bca055b095673620d8bb4c98ef6c610a22f999294ed11ae35d43621e93-Picture3.png)

![Picture4.png](https://pic.leetcode-cn.com/f75d89219ad93c69757b187c64784b4c7a57dce7911884fe82f14073d654d32f-Picture4.png)

```
one = one ^ n & ~two
two = two ^ n & ~one
以上是对数字的二进制中 “一位” 的分析，而 int 类型的其他 31 位具有相同的运算规则，因此可将以上公式直接套用在 32 位数上。
```



```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ones = 0, twos = 0;
        for(auto num :nums){
            ones = ones^num&(~twos);
            twos = twos^num&(~ones);
        }

        return ones;

    }
};
```



# 排序

#### [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

```
输入: [10,2]
输出: "102"
输入: [3,30,34,5,9]
输出: "3033459"
0 < nums.length <= 100
```

第一遍自己写的时候没有解决，参考了其他相关代码。

此题求拼接起来的最小数字，本质上是一个排序问题。设数组 nums中任意两数字的字符串为 x和 y ，则规定 排序判断规则 为：

若拼接字符串 x + y > y + x ，则 x “大于” y ；
反之，若 x + y < y + x ，则 x“小于” y ；

![Picture1.png](https://pic.leetcode-cn.com/95e81dbccc44f26292d88c509afd68204a86b37d342f83d109fa7aa0cd4a6049-Picture1.png)

- [ ] ```cpp
  class Solution {
  public:
      string minNumber(vector<int>& nums) {
          vector<string>strs;
          for(int i = 0; i<nums.size(); i++){
              strs.emplace_back(to_string(nums[i]));
          }
          // quickSort(strs,0,strs.size()-1);
          sort(strs.begin(),strs.end(),[](string &x,string &y){return x+y<y+x;});
          string res;
          for(auto ch:strs){
              res+=ch;
          }
          return res;
      }
  private:
      void quickSort(vector<string>& strs, int l, int r) {
          if(l>=r)return;
          int i = l-1, j= r+1;
          while(i<j){
              do i++;
              while(strs[i]+strs[l]<strs[l]+strs[i]);
              do j--;
              while(strs[j]+strs[l]>strs[l]+strs[j]);
              if(i<j)swap(strs[i],strs[j]);
  
          }
          quickSort(strs,l,j);
          quickSort(strs,j+1,r);
      }
  };
  ```

看到的其他版本

```cpp
class Solution {
public:
    static bool cmp(int a,int b)
    {
        string s1=to_string(a);
        string s2=to_string(b);
        return s1+s2<s2+s1;
    }
    string minNumber(vector<int>& nums) 
    {
        sort(nums.begin(),nums.end(),cmp);
        string res="";
        for(int i=0;i<nums.size();i++)res+=to_string(nums[i]);
        return res;
    }
};

```

#### [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

```
输入: [1,2,3,4,5]
输出: True
输入: [0,0,1,2,5]
输出: True
限制：

数组长度为 5 

数组的数取值为 [0, 13] .
```

本题解法非常巧妙：

因为是只抽出五张牌,那么要保证五张牌是顺子：

1、除了大小王之外，其他牌是无重复的。

2、可以假设5张牌中最大的为max,最小的牌为min(大小王除外)。满足下列等式即可：

max-min<5;

![Picture1.png](https://pic.leetcode-cn.com/df03847e2d04a3fcb5649541d4b6733fb2cb0d9293c3433823e04935826c33ef-Picture1.png)

方法1：排序加遍历的方法

1、先对数组进行排序。

2、统计大小王（0）的个数、

3、获取最大或者最小的牌：排序完后，nums[4]为最大的牌；nums[zero]为最小额的牌，zero为大小王个数。

只要满足 最大-最小<5既可以实现目标。

```c++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        int zeroNum = 0;
        for(int i = 1; i<5;i++){
            if(nums[i-1]==0)zeroNum+=1;
            else if(nums[i-1]==nums[i])return false;
        }
        return nums[4]-nums[zeroNum]<5;

    }
};
```

方法2：哈希set+遍历

遍历五张牌，大小王（0）跳过

判重复使用set即可

获取最大最小牌。

```c++
unordered_set<int>set1;
        int max1 = 0, min1 = 14;
        for(auto num:nums){
            if(num==0)continue;
            max1 = max(max1,num);
            min1 = min(min1,num);
            if(set1.find(num)!=set1.end())return false;
            set1.insert(num);
        }
        return max1-min1<5;
```



#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

难度中等1126

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

 

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

 

**提示：**

- `1 <= nums.length <= 105`
- `k` 的取值范围是 `[1, 数组中不相同的元素的个数]`
- 题目数据保证答案唯一，换句话说，数组中前 `k` 个高频元素的集合是唯一的

 

**进阶：**你所设计算法的时间复杂度 **必须** 优于 `O(n log n)` ，其中 `n` 是数组大小。

利用哈希表存储数的数值，对应频次。然后放入vector<pair<int,int>>中，对频次排序，存储前k个值即可。

```c++
bool cmp(pair<int,int>a,pair<int,int>b){
       return a.second<b.second;
   }
class Solution {
   
public:
    
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int>hashmap;
        for(auto num:nums){
            hashmap[num]++;
        }
        vector<pair<int,int>>vec;
        for(auto p:hashmap){
            vec.emplace_back(p);
        }
        sort(vec.begin(),vec.end(),cmp);
        vector<int>res;
        while(k--){
            res.emplace_back(vec.back().first);
            vec.pop_back();
        }
        return res;
    }
};
```

利用小根堆维护前k频次的数

```c++
 struct tmp{
    bool operator()(pair<int,int>&a, pair<int,int>&b){
        return a.second > b.second;
    }
 };
 
class Solution {
    
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int>hash;
        for(auto num:nums)hash[num]++;
        //建立小顶堆 并且按照第二个值排序
        priority_queue<pair<int,int>,vector<pair<int,int>>,tmp>que;
        for(auto &[num,count]:hash){
            if(que.size() == k){
                if(que.top().second<count){
                    que.pop();
                    que.emplace(num,count);
                }
            }else{
                que.emplace(num,count);
            }
        }
        vector<int>res;
        while(!que.empty()){
            res.emplace_back(que.top().first);
            que.pop();
        }
        return res;

    }
};
```



#### [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

- `0 <= k <= arr.length <= 10000`
- `0 <= arr[i] <= 10000`

本题较简单，找最小的K个数。直接先排序，将前k个数存入res数组即可。

方法1：

用封装好的sort直接排序。

```c++
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int>res;
        if(k== 0 )return {};
        sort(arr.begin(),arr.end());
        for(int i = 0; i<k; i++){
            res.emplace_back(arr[i]);
        }
        return res;

    }
};
```

方法2：堆排序

```cpp
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int>res(k,0);
        priority_queue<int>que;
        if(k == 0)return res;
        //维持一个有k个数字的大根堆
        for(int i = 0; i<k; i++){
            que.push(arr[i]);
        }
        for(int i = k; i<arr.size(); i++){
            if(que.top()>arr[i]){
                que.pop();
                que.push(arr[i]);
            }
        }
        for(int i = 0; i<k; i++){
            res[i] = que.top();
            que.pop();
        }
        return res;
    }
};
```



#### [剑指 Offer 41. 数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。

```
输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]
输入：
["MedianFinder","addNum","findMedian","addNum","findMedian"]
[[],[2],[],[3],[]]
输出：[null,null,2.00000,null,2.50000]

```

本题第一次尝试做的实现想的是，每操作一次插入数，重新进行排序，然后判断数列中数的个数，求取中位数。这样时间复杂度太高了O(n2logn);

下面题解采用的是利用小根堆和大根堆来维护中位数.

建立一个大根堆和小根堆，各自保存一半元素。

queMax为小根堆：保存大于中位数的元素，queMin为大根堆：保存小于中位数的元素。各自保存一半元素，动态调整两个堆的元素数量。

如果大根堆数量比小根堆多1，返回大根堆堆顶，否则返回两个堆顶均值。

![Picture1.png](https://pic.leetcode-cn.com/25837f1b195e56de20587a4ed97d9571463aa611789e768914638902add351f4-Picture1.png)



```c++
class MedianFinder {
public:
    priority_queue<int,vector<int>,greater<int>>queMax;//小根堆
    priority_queue<int,vector<int>,less<int>>queMin;//大根堆
    /** initialize your data structure here. */
    MedianFinder() {

    }
    
    void addNum(int num) {
        if(queMin.empty() || num<=queMin.top()){
            queMin.push(num);
            //判断两个堆大小是否相差1
            if(queMax.size()+1<queMin.size()){
                queMax.push(queMin.top());
                queMin.pop();
            }
        }else{
            queMax.push(num);
            //判断
            if(queMax.size()>queMin.size()){
                queMin.push(queMax.top());
                queMax.pop();
            }
        }

    }
    
    double findMedian() {
        if(queMin.size()>queMax.size())return queMin.top();
        return (queMin.top()+queMax.top())/2.0;

    }
};
```



# 字符串

## 1、KMP算法

KMP算法是一种**字符串匹配**算法，可以在 O(n+m) 的时间复杂度内实现两个字符串的匹配。

![img](https://pic1.zhimg.com/80/v2-2967e415f490e03a2a9400a92b185310_720w.jpg?source=1940ef5c)

### 1.1朴素算法Brute-Force

从前往后

- 枚举 i = 0, 1, 2 … , len(S)-len(P)
- 将 S[i : i+len(P)] 与 P 作比较。如果一致，则找到了一个匹配。

![img](https://pica.zhimg.com/80/v2-1892c7f6bee02e0fc7baf22aaef7151f_720w.jpg?source=1940ef5c)

![img](https://pic1.zhimg.com/80/v2-36589bc0279263ec8641a295aea66a0c_720w.jpg?source=1940ef5c)

![img](https://pic1.zhimg.com/80/v2-ed28c8d60516720cc38c48d135091a58_720w.jpg?source=1940ef5c)

暴力匹配每次只移动一位，所以最坏的匹配情况如下：

![img](https://pic1.zhimg.com/80/v2-4fe5612ff13a6286e1a8e50a0b06cd96_720w.jpg?source=1940ef5c)

比较的趟数非常多，那么怎么才能减小比较的次数呢。尽可能利用残余信息。

在 Brute-Force 中，如果从 S[i] 开始的那一趟比较失败了，算法会直接开始尝试从 **S[i+1] 开始比较**。这种行为，属于典型的“**没有从之前的错误中学到东西”**。我们应当注意到，一次失败的匹配，会给我们提供宝贵的信息——如果 S[i : i+len(P)] 与 P 的匹配是在第 r 个位置失败的，那么从 S[i] 开始的 (r-1) 个连续字符，一定与 P 的前 (r-1) 个字符一模一样！

![img](https://pic1.zhimg.com/80/v2-7dc61b0836af61e302d9474eeeecfe83_720w.jpg?source=1940ef5c)

我们需要跳过那些不可能匹配的字符串。

有些趟字符串比较是有**可能会成功的**；有些则毫无可能。我们刚刚提到过，优化 Brute-Force 的路线是“**尽量减少比较的趟数**”，而如果我们跳过那些**绝不可能成功的**字符串比较，则可以希望复杂度降低到能接受的范围。

以下的例子，有些字符串是肯定不能匹配的。

![img](https://pic2.zhimg.com/80/v2-372dc6c567ba53a1e4559fdb0cb6b206_720w.jpg?source=1940ef5c)

首先，利用上一节的结论。既然是在 P[5] 失配的，那么**说明 S[0:5] 等于** P[0:5]，即”abcab”. 现在我们来考虑：从 S[1]、S[2]、S[3] 开始的匹配尝试，有没有可能成功？

从 S[1] 开始肯定没办法成功，因为 S[1] = P[1] = ‘b’，和 P[0] 并不相等。从 S[2] 开始也是没戏的，因为 S[2] = P[2] = ‘c’，并不等于P[0]. 但是从 S[3] 开始是有可能成功的——至少按照已知的信息，我们推不出矛盾

![img](https://pic2.zhimg.com/80/v2-67dd66b86323d3d08f976589cf712a1a_720w.jpg?source=1940ef5c)

带着“跳过不可能成功的尝试”的思想，我们来看[next数组](https://www.zhihu.com/search?q=next数组&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1032665486})。

### 1.2 next数组

next数组是对于模式串而言的。P 的 [next 数组](https://www.zhihu.com/search?q=next+数组&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1032665486})定义为：next[i] 表示 P[0] ~ P[i] 这一个子串，使得 **前k个字符**恰等于**后k个字符** 的最大的k. 特别地，k不能取i+1（因为这个子串一共才 i+1 个字符，自己肯定与自己相等，就没有意义了）。

![img](https://pic2.zhimg.com/80/v2-49c7168b5184cc1744459f325e426a4a_720w.jpg?source=1940ef5c)

把模式串作为一个标尺，在主串上进行移动，BF算法是每失配之后只往后移动一位；那么改进算法是失配后，移动多位，下面看看如何确定跳过多少位。

![preview](https://pic2.zhimg.com/v2-d6c6d433813595dce5aad08b40dc0b72_r.jpg?source=1940ef5c)

next数组的性质是：p【0，i】这一段模式串中，前next[i]个字符和后next[i]个字符相同。失配在 P[r], 那么P[0]~P[r-1]这一段里面，**前next[r-1]个字符恰好和后next[r-1]个字符相等**——也就是说，我们可以拿长度为 next[r-1] 的那一段前缀，来顶替当前后缀的位置，让匹配继续下去！

![img](https://pic2.zhimg.com/80/v2-6ddb50d021e9fa660b5add8ea225383a_720w.jpg?source=1940ef5c)

如上图所示，绿色部分是成功匹配，失配于红色部分。深绿色手绘线条标出了相等的前缀和后缀，**其长度为next[右端]**. 由于手绘线条部分的字符是一样的，所以直接把前面那条移到后面那条的位置。因此说，**next数组为我们如何移动标尺提供了依据**。接下来，我们实现这个优化的算法。

代码实现：

分为两个部分：建立next数组、利用next数组进行匹配。首先是建立next数组。

```cpp
void getNext(int* next, const string& s){
    int j = -1;
    next[0] = j;
    for(int i = 1; i < s.size(); i++) { // 注意i从1开始
        while (j >= 0 && s[i] != s[j + 1]) { // 前后缀不相同了
            j = next[j]; // 向前回退
        }
        if (s[i] == s[j + 1]) { // 找到相同的前后缀
            j++;
        }
        next[i] = j; // 将j（前缀的长度）赋给next[i]
    }
}
```

下面用题目来应用一下KMP字符串匹配

#### [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

说明：

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 needle 是空字符串时我们应当返回 0 。

```
输入：haystack = "hello", needle = "ll"
输出：2
```

```
输入：haystack = "aaaaa", needle = "bba"
输出：-1
```

```
输入：haystack = "", needle = ""
输出：0


```

```cpp
class Solution {
public:
    void getNext(int *next, const string &s){
        int j = -1;
        next[0] = j;
        for(int i = 1; i<s.size(); i++){
            //字符串失配
            while(j>=0 && s[i]!=s[j+1]){
                j = next[j];
            }
            if(s[i] == s[j+1])j++;
            next[i] = j;
        }
    }
    int strStr(string haystack, string needle) {
        if(needle.size()==0)return 0;
        int next[needle.size()];
        getNext(next,needle);
        int j = -1;
        for(int i = 0; i < haystack.size(); i++){
            while(j>=0 && haystack[i] != needle[j+1])j = next[j];
            if(haystack[i] == needle[j+1])j++;
            if(j == (needle.size()-1))return (i-needle.size()+1);
        }

        return -1;

    }
};
```

#### [459. 重复的子字符串](https://leetcode-cn.com/problems/repeated-substring-pattern/)

给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。

```
输入: "abab"

输出: True

解释: 可由子字符串 "ab" 重复两次构成。
```

```
输入: "aba"

输出: False
```

```
输入: "abcabcabcabc"

输出: True
```

```c++
class Solusion{
 public:
    void getNext(int *next, const string& s){
        int j = -1;
        next[0] = -1;
        for(int i = 1; i<s.size(); i++){
			while(j>0 && s[i]!=s[j+1]){
				j = next[j];
            }
            if(s[i] == s[j])j++;
            next[i] = j;
        }
    }
    bool repeatedSubstringPattern(string s) {
        if(s.size() == 0)return 0;
        int next[s.size()];
        getNext(next,s);
        //怎么判断是否有重复字串  
        //非重复串时next全为-1 重复串后数字会递增
        int size = s.size() - next[s.size()-1]-1;
        if(next[s.size()-1]!=-1 && s.size()%size==0)return true;
        return false;
    
}
```

# 哈希

### [剑指 Offer II 010. 和为 k 的子数组](https://leetcode-cn.com/problems/QTMn0o/)

给定一个整数数组和一个整数 `k` **，**请找到该数组中和为 `k` 的连续子数组的个数。

```
输入:nums = [1,1,1], k = 2
输出: 2
解释: 此题 [1,1] 与 [1,1] 为两种不同的情况
输入:nums = [1,2,3], k = 3
输出: 2
1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107
```

本题中的k可以为负数，如果采用滑动窗口，没办法判断head移动到什么位置。本题采用哈希表+前缀和进行优化。

思路1：暴力解法

直接枚举，考虑以i结尾，0<=j<=i<nums.size().找到[j,i]区间和为k的子数组。



```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int count = 0;
        for (int start = 0; start < nums.size(); ++start) {
            int sum = 0;
            for (int end = start; end >= 0; --end) {
                sum += nums[end];
                if (sum == k) {
                    count++;
                }
            }
        }
        return count;
    }
};


```

但是提交的时候发现超时了。

考虑方法2：前缀和 + 哈希表优化

方法1中对于i，需要枚举所有的j来判断是否符合条件。

![image-20220121220003842](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220121220003842.png)

```c++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int,int>hash;
        hash[0] = 1;
        int count = 0,sum = 0;
        for(auto &x:nums){
            sum+=x;
            if(hash.find(sum-k)!=hash.end())count+=hash[sum-k];
            hash[sum]++;
        }
        return count;
         
    }
};
```

### [剑指 Offer II 011. 0 和 1 个数相同的子数组](https://leetcode-cn.com/problems/A1NYOS/)

给定一个二进制数组 `nums` , 找到含有相同数量的 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。

```
输入: nums = [0,1]
输出: 2
说明: [0, 1] 是具有相同数量 0 和 1 的最长连续子数组。
输入: nums = [0,1,0]
输出: 2
说明: [0, 1] (或 [1, 0]) 是具有相同数量 0 和 1 的最长连续子数组。
1 <= nums.length <= 105
nums[i] 不是 0 就是 1

```

本题思路进行转换以下就可以变成和为k的子数组相同的题。

求数组nums中含有相同数量的0和1的最长连续子数组，由于「0和 1 的数量相同」等价于「1 的数量减去 0 的数量等于 0」，我们可以将数组中的 0视作 -1，则原问题转换成「求最长的连续子数组，其元素和为 0」。

那么可以将nums数组转化成一个长度相同的newnums，当nums中的元素为0视为-1,1视为1。前缀和为0的即满足要求。

```c++
/*
前缀和：
     * -[0,0,1,0,0,0,1,1]
     * - i               preSum = -1, (用 -1 替换 0);
     * -[0,0,1,0,0,0,1,1]
     * -   i             preSum = -2
     * -[0,0,1,0,0,0,1,1]
     * -     i           preSum = -1
     * -[0,0,1,0,0,0,1,1]
     * -       i         preSum = -2
     * -[0,0,1,0,0,0,1,1]
     * -         i       preSum = -3
     * -[0,0,1,0,0,0,1,1]
     * -           i     preSum = -4
     * -[0,0,1,0,0,0,1,1]
     * -             i   preSum = -3
     * -[0,0,1,0,0,0,1,1]
     * -               i preSum = -2
     * 观察可以发现, 当前缀和相同时, 前一个 i1 后面一个位置开始一直到 i2 的区间是满足题目要求的子数组, 即 nums[i1+1...i2] 满足题
     * 目要求, 并且 i2 - i1 = 子数组长度, 所以我们只需要计算出 nums[0...n-1] 每一个位置的前缀和, 一旦发现当前的计算出的前缀和在
     * 之前已经出现过, 就用当前的索引 i2 - 之前的索引 i1 即可求出本次得到的子数组长度,。因为需要求得的是最长连续子数组，所以应用一
     * 个变量 maxLength 来保存每一次计算出的子数组长度, 取较大值。也因为, 我们需要保存每一个位置的前缀和, 并且还需要通过前缀和找到
     * 相应位置的索引, 所以，使用 HashMap 来存放 {前缀和:索引}, 在上面例子中我们通过观察得到了 i2 - i1 = 数组长度, 但是有一个很隐
     * 蔽的缺陷, 即当整个数组即为答案时, i2 = nums.length - 1, i1 = 0 此时得到的数组长度为 nums.length - 1 这显然是错误的。因此
     * , 为了避免这个错误, 我们初始将 Map 中添加一对 {前缀和:索引}, 即 put(0,-1), 0代表前一个不存在的元素前缀和为 0, -1 代表不存
     * 在元素的索引。
     * 当定义了这些条件后, 我们开始用指针 i 遍历数组nums[0...nums.length - 1] 位置上的每一个元素。
     * 一、用变量 sum 来纪录[0...i]区间的和:
     * -   1.当 nums[i] == 0 时, sum += -1
     * -   2.当 nums[i] == 1 时, sum += 1
     * 二、接着判断 sum 是否已经存在于 HashMap 中:
     * -   1. 如果存在, 则取出 sum 所对应的索引 j, 那么 nums[j+1,i] 就是一个满足答案的子区间, 用
     * -      maxLength = Math.max(maxLengnth, i - j); 来纪录最长子数组。
     * -   2. 如果不存在, 则将 {sum:i} 存放在 HashMap 中作为纪录。
     * 当数组遍历完毕时, maxLength 中保存的即为答案数组的长度。
     * <p>
     */
```

代码如下

```c++
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        unordered_map<int,int>hash;
        hash[0] = -1;
        int res = 0,sum = 0;
        for(int i = 0; i<nums.size(); i++){
            int num = nums[i];
            if(num == 0){
                sum-=1;
            }else if(num == 1){
                sum+=1;
            }
            if(hash.count(sum)){
                int preIndex = hash[sum];
                res = max(res,i-preIndex);
            }else{
                hash[sum] = i;
            }
            
        }
        return res;

    }
};
```

在滑动窗口类型的问题中都会有两个指针，一个用于「延伸」现有窗口的 r 指针，和一个用于「收缩」窗口的 l指针。在任意时刻，只有一个指针运动，而另一个保持静止。



### [剑指 Offer II 014. 字符串中的变位词](https://leetcode-cn.com/problems/MPnaiL/)

给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1` 的某个变位词。

换句话说，第一个字符串的排列之一是第二个字符串的 **子串** 。

```
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
输入: s1= "ab" s2 = "eidboaoo"
输出: False
1 <= s1.length, s2.length <= 104
s1 和 s2 仅包含小写字母

```

方法1 滑动窗口

当两个字符串每个字符个数相等时，一个字符串才是另一个字符串的排列。我们通过维护一个固定窗口维护cnt2,滑动窗口每右移一次，统计进入窗口的字符，减去离开窗口的字符。判断cnt1和cnt2是否相等。

```c++
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        int m = s1.size(),n = s2.size();
        vector<int>cnt1(26),cnt2(26);
        if(m>n)return false;

        for(int i = 0; i<m; i++){
            cnt1[s1[i] - 'a']++;
            cnt2[s2[i] - 'a']++;
        }
        if(cnt1==cnt2)return true;

        for(int i = m; i<n; i++){
            cnt2[s2[i] - 'a']++;
            cnt2[s2[i-m] - 'a']--;
            if(cnt2 == cnt1)return true;
        }
        return false;

    }
};
```

优化，滑动窗口每次只需要进出一个字符，但是每次都比较cnt1和cnt2两个数组。其实可以用diff记录cnt1和cnt2的不同值的个数，转化为判断diff是否为0即可。滑动窗口分别记录进入字符x和出去字符y。

x=y则对cnt2没有影响，

x！=y，对于字符 x，在修改 cnt2之前若有 cnt2[x]=cnt1[x]，则将diff 加一；在修改cnt2之后若有 cnt2[x]=cnt1[x]，则将 diff 减一。字符 y同理。



### [剑指 Offer II 016. 不含重复字符的最长子字符串](https://leetcode-cn.com/problems/wtcaE1/)

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长连续子字符串** 的长度。

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子字符串是 "abc"，所以其长度为 3
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子字符串是 "b"，所以其长度为 1。
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
0 <= s.length <= 5 * 104
s 由英文字母、数字、符号和空格组成

```

本题采用滑动窗口来解决。

使用两个指针表示字符串中的子串窗口，

```c++
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        int m = s1.size(),n = s2.size();
        vector<int>cnt(26);
        int diff = 0;
        if(m>n)return false;

        for(int i = 0; i<m; i++){
            --cnt[s1[i] - 'a'];
            ++cnt[s2[i] - 'a'];
        }
        //记录两个数组的差异
        for(int c: cnt){
            if(c!=0){
                ++diff;
            }
        }

        if(diff == 0)return true;
        for(int i =m ;i<n; i++){
            int x = s2[i] -'a', y =s2[i-m] - 'a';//分别表示进出数组的元素
            if(x == y)continue;
            if(cnt[x] == 0){
                diff++;
            }
            ++cnt[x];
            if(cnt[x] == 0){
                --diff;
            }
            if(cnt[y] == 0){
                ++diff;
            }
            --cnt[y];
            if(cnt[y] == 0)--diff;
            if(diff == 0)return true;
        }
        return false;


    }
};
```

### [剑指 Offer II 015. 字符串中的所有变位词](https://leetcode-cn.com/problems/VabMRr/)

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **变位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

```
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的变位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的变位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的变位词。
1 <= s.length, p.length <= 3 * 104
s 和 p 仅包含小写字母


```

### [5268. 找出两数组的不同](https://leetcode-cn.com/problems/find-the-difference-of-two-arrays/)

给你两个下标从 0 开始的整数数组 nums1 和 nums2 ，请你返回一个长度为 2 的列表 answer ，其中：

answer[0] 是 nums1 中所有 不 存在于 nums2 中的 不同 整数组成的列表。
answer[1] 是 nums2 中所有 不 存在于 nums1 中的 不同 整数组成的列表。
注意：列表中的整数可以按 任意 顺序返回。

```
输入：nums1 = [1,2,3], nums2 = [2,4,6]
输出：[[1,3],[4,6]]
解释：
对于 nums1 ，nums1[1] = 2 出现在 nums2 中下标 0 处，然而 nums1[0] = 1 和 nums1[2] = 3 没有出现在 nums2 中。因此，answer[0] = [1,3]。
对于 nums2 ，nums2[0] = 2 出现在 nums1 中下标 1 处，然而 nums2[1] = 4 和 nums2[2] = 6 没有出现在 nums2 中。因此，answer[1] = [4,6]。

输入：nums1 = [1,2,3,3], nums2 = [1,1,2,2]
输出：[[3],[]]
解释：
对于 nums1 ，nums1[2] 和 nums1[3] 没有出现在 nums2 中。由于 nums1[2] == nums1[3] ，二者的值只需要在 answer[0] 中出现一次，故 answer[0] = [3]。
nums2 中的每个整数都在 nums1 中出现，因此，answer[1] = [] 。 


```

利用set进行去重

```c++
class Solution {
public:
    vector<vector<int>> findDifference(vector<int>& nums1, vector<int>& nums2) {
        //利用set去除重复
        set<int>hash1,hash2;
        vector<vector<int>>res;
        for(int num1:nums1){
            hash1.insert(num1);
        }
        for(int num2 : nums2){
            hash2.insert(num2);
            if(hash1.count(num2))hash1.erase(num2);
        }
        for(int num1:nums1){
            if(hash2.count(num1))hash2.erase(num1);
        }
        res.emplace_back(vector<int>(hash1.begin(),hash1.end()));
        res.emplace_back(vector<int>(hash2.begin(),hash2.end()));
        return res;

    }
};
```



#### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
0 <= nums.length <= 105
-109 <= nums[i] <= 109
```

方法1 利用哈希结合存储数组中的元素，并进行去重

然后先查找当前数字-1的值，在逐个查找+1的值

```c++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        //朴素哈希算法
        unordered_set<int>hashset;
        for(auto num:nums)hashset.insert(num);
        int ans = 0;
        for(auto num:hashset){
            int cur = num;
            if(!hashset.count(cur-1)){
                while(hashset.count(cur+1))cur++;
            }
            ans = max(ans,cur-num+1);
        }
        return ans;
    }
};
```



# 双指针算法



### [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

```
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
```

```
输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
```

- `0 <= nums.length <= 100`
- `0 <= nums[i] <= 50`
- `0 <= val <= 100`

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int fastIndex = 0,slowIndex = 0;
        for(;fastIndex<nums.size(); fastIndex++){
            if(nums[fastIndex]!=val)
                nums[slowIndex++]=nums[fastIndex];
     
        }
        return slowIndex;

    }
};
```



### [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

```
输入：s = "We are happy."
输出："We%20are%20happy."
```

```cpp
class Solution {
public:
    string replaceSpace(string s) {
        int spaceNum = 0;
        int oldSize = s.size()-1;
        for(auto ch:s){
            if(ch==' ')spaceNum++;
        }
        s.resize(s.size()+spaceNum*2);
        int newSize = s.size()-1;
        for(int i = newSize,j = oldSize; i>j; i--,j--){
            if(s[j] == ' '){
                s[i] = '0';
                s[i-1] = '2';
                s[i-2] = '%';
                i-=2;

            }else{
                s[i] = s[j];
            }
        }
        return s;


    }
};
```

### [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

给你一个字符串 s ，逐个翻转字符串中的所有 单词 。

单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。

请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。

- 输入字符串 `s` 可以在前面、后面或者单词间包含多余的空格。
- 翻转后单词间应当仅用一个空格分隔。
- 翻转后的字符串中不应包含额外的空格。

```
输入：s = "the sky is blue"
输出："blue is sky the"
```

```
输入：s = "  hello world  "
输出："world hello"
```

```
输入：s = "a good   example"
输出："example good a"
```

本题很重要的一点是要先去除多余的空格字符：字符串最前面的空格 中间的空格 字符串最后的空格。

然后再考虑字符串翻转的问题：

```c++
class Solution {
public:
    // void removespace(string &s){
    //     for(int i = s.size(); i>0; i--){
    //         if(s[i] == s[i-1] && s[i]==' ')
    //             s.erase(s.begin()+i);
    //     }
    //     //删除最后面的空格
    //     if(s.size()>0 && s[s.size()-1] == ' '){
    //         s.erase(s.begin()+s.size()-1);
    //     }
    //     //删除最前面的空格
    //     if(s.size()>0 && s[0]==' ')
    //         s.erase(s.begin());


    // }
    //双指针删除空格
    void removespace(string &s){
        //删除前面的空格
        int fastIndex = 0, slowIndex = 0;
        while(s.size()>0 && s[fastIndex]==' ' && fastIndex<s.size())fastIndex++;
        //删除中间的空格
        for(; fastIndex<s.size(); fastIndex++){
            if(fastIndex-1>0 &&
            s[fastIndex-1]==s[fastIndex]&&
            s[fastIndex]==' ')continue;
            else{
                s[slowIndex++] = s[fastIndex];
            }
        }
        if(slowIndex-1>0 && s[slowIndex-1] == ' '){
            s.resize(slowIndex-1);
        }else{
            s.resize(slowIndex);
        }

    }
    string reverseWords(string s) {
        removespace(s);
        reverse(s.begin(),s.end());
        int j = 0;
        for(int i = 0; i<s.size(); i++){
            if(s[i] == ' ')
            {
                reverse(s.begin()+j,s.begin()+i);
                j = i+1;

            }
                
        }
        reverse(s.begin()+j,s.end());

        return s;


    }
};
```

### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。



![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)



```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex2.jpg)

```
输入：head = [1,2]
输出：[2,1]
```

```
输入：head = []
输出：[]
```

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *cur = head;
        ListNode *pre = NULL;
        ListNode *temp = cur;
        while(cur){
            temp = cur->next;
            cur->next = pre;
            //更新操作
            pre = cur;
            cur = temp;
            
        }
        return pre;

    }
};
```

### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点

![img](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

```
输入：head = [1], n = 1
输出：[]
```

```
输入：head = [1,2], n = 1
输出：[1]
```

本题依然属于双指针的应用，要求删除倒数第n个节点，fast先走n步，slow和fast同时移动，fast到链表末尾时，删除slow指向的节点即可。

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode *preHead = new ListNode(-1);
        preHead->next =  head;
        ListNode *fast = preHead;
        ListNode *slow = preHead;
        while(n-- && fast!=nullptr){
            fast =fast->next;
        }
        fast = fast->next;
        while(fast){
            fast =fast->next;
            slow = slow->next;
        }

        slow->next = slow->next->next;
        return preHead->next;

    }
};
```

### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

图示两个链表在节点 c1 开始相交：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

题目数据 **保证** 整个链式结构中不存在环。

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。

![img](https://assets.leetcode.com/uploads/2021/03/05/160_example_1_1.png)

输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'

![img](https://assets.leetcode.com/uploads/2021/03/05/160_example_2.png)

输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_3.png)

输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        int size1=0,size2=0;
        int gap = 0;
        ListNode* cur1 = headA;
        ListNode* cur2 = headB;
        while(cur1){
            cur1 = cur1->next;
            size1++;
        }
        while(cur2){
            cur2 =cur2->next;
            size2++;
        }
        cur1 = headA;
        cur2 = headB;
        if(size1<size2){
            swap(size1,size2);
            swap(cur1,cur2);
        }
        gap = size1 - size2;
        while(gap--){
            cur1 = cur1->next;
        }
        while(cur1!=NULL){
            if(cur1 == cur2)return cur1;
            else{
                cur1 = cur1->next;
                cur2 = cur2->next;
            }
        }
        return NULL;

    }
};
```

求两个节点的交点，即链表指针相同的点。

![面试题02.07.链表相交_1](https://code-thinking.cdn.bcebos.com/pics/%E9%9D%A2%E8%AF%95%E9%A2%9802.07.%E9%93%BE%E8%A1%A8%E7%9B%B8%E4%BA%A4_1.png)

我们求出两个链表的长度，并求出两个链表长度的差值，然后让curA移动到，和curB 末尾对齐的位置，如图：

![面试题02.07.链表相交_2](https://code-thinking.cdn.bcebos.com/pics/%E9%9D%A2%E8%AF%95%E9%A2%9802.07.%E9%93%BE%E8%A1%A8%E7%9B%B8%E4%BA%A4_2.png)

此时我们就可以比较curA和curB是否相同，如果不相同，同时向后移动curA和curB，如果遇到curA == curB，则找到交点。

否则循环退出返回空指针。

方法二：只有当链表 headA 和 headB 都不为空时，两个链表才可能相交。因此首先判断链表headA 和 headB 是否为空，如果其中至少有一个链表为空，则两个链表一定不相交，返回 null。

当链表headA和headB都不为空时，创建两个指针pa和pb，遍历两个链表的每个节点。

1、每步操作更新指针pa和pb。若都不为空指向下一个节点。

2、如果pa为空，则让指针pa指向headb的头结点，pb为空，让指针pb指向heada的头结点。

3、当pa和pb指向同一个节点时，或者都为空时，返回它们的节点或者null.



```c++
 ListNode * pA = headA; ListNode * pB = headB;
        if(headA == nullptr || headB ==nullptr)return nullptr;
        while(pA!=pB){
            pA = pA ==nullptr ? headB:pA->next;
            pB = pB == nullptr? headA:pB->next; 
        }
        return pA;
```

下图为详细解释

![相交链表.png](https://pic.leetcode-cn.com/e86e947c8b87ac723b9c858cd3834f9a93bcc6c5e884e41117ab803d205ef662-%E7%9B%B8%E4%BA%A4%E9%93%BE%E8%A1%A8.png)

和判断有环类似。

### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

不允许修改 链表。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

```
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点
```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

```
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *fast = head;
        ListNode *slow = head;
        while(fast!=NULL && fast->next!=NULL){
            fast = fast->next->next;
            slow = slow->next;
            //快慢指针相遇
            if(fast == slow){
                ListNode *cur1 = slow;
                ListNode *cur2 = head;
                while(cur1 != cur2){
                    cur1 = cur1->next;
                    cur2 =cur2->next;
                }
                return cur1;
            }
        }
        return NULL;
        
    }
};
```

### [剑指 Offer II 008. 和大于等于 target 的最短子数组](https://leetcode-cn.com/problems/2VG8Kg/)

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

```
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
输入：target = 4, nums = [1,4,4]
输出：1
输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0
```

方法1： 双指针模拟滑动窗口

定义指针left 和right 为窗口的左右起点。维护变量sum为窗口内的数组元素和。

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0,right = 0;
        int res[nums.size()];
        int minlenth = 0;
        int sum = 0;
        for(int i = 1; i<nums.size();i++){
            s[i] = s[i-1] + nums[i];
        }
        for(;right<nums.size(); right++){
            if(s[right] - s[left] >= target)left++;
        }

    }
};
```

方法2 前缀和加二分查找

创建前缀和数组sums，由于数组中没有负数，所以前缀和数组是单调不减的数组。

### [713. 乘积小于K的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/)

给定一个正整数数组 `nums`和整数 `k` 。

请找出该数组内乘积小于 `k` 的连续的子数组的个数。

```
输入: nums = [10,5,2,6], k = 100
输出: 8
解释: 8个乘积小于100的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于100的子数组。
输入: nums = [1,2,3], k = 0
输出: 0
1 <= nums.length <= 3 * 104
1 <= nums[i] <= 1000
0 <= k <= 106 
```

双指针算法（滑动窗口）

对于维护乘积、和等大于等于或者小于等于某个值的题，其中要求连续数组的个数，都可以采用滑动窗口，滑动窗口大部分都用双指针来实现的。

```c++
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int sum =1;
        int left = 0,res = 0;
        if(k<=1)return 0;

        for(int right = 0; right<nums.size();right++){
            sum*= nums[right];

            while(sum>= k){
                sum/= nums[left];
                left++;
            }
            res += right -left+1;


        }
        return res;

    }
};
```



# 深度优先搜索和回溯算法

![回溯算法大纲](https://img-blog.csdnimg.cn/20210219192050666.png)

回溯是递归的副产品，只要有递归就会有回溯。

**因为回溯的本质是穷举，穷举所有可能，然后选出我们想要的答案**，如果想让回溯法高效一些，可以加一些剪枝的操作，但也改不了回溯法就是穷举的本质。

回溯法，一般可以解决如下几种问题：

- 组合问题：N个数里面按一定规则找出k个数的集合
- 切割问题：一个字符串按一定规则有几种切割方式
- 子集问题：一个N个数的集合里有多少符合条件的子集
- 排列问题：N个数按一定规则全排列，有几种排列方式
- 棋盘问题：N皇后，解数独等等

## 如何理解回溯法

**回溯法解决的问题都可以抽象为树形结构**，是的，我指的是所有回溯法的问题都可以抽象为树形结构！

因为回溯法解决的都是在集合中递归查找子集，**集合的大小就构成了树的宽度，递归的深度，都构成的树的深度**。

递归就要有终止条件，所以必然是一颗高度有限的树（N叉树）。



#### [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

```
输入：m = 2, n = 3, k = 1
输出：3
```

```
输入：m = 3, n = 1, k = 0
输出：1
```

- `1 <= n,m <= 100`
- `0 <= k <= 20`

方法一 ：广度优先搜索

 两者目标都是遍历整个矩阵，不同点在于搜索顺序不同。DFS 是朝一个方向走到底，再回退，以此类推；BFS 则是按照“平推”的方式向前搜索。

通常采用队列实现广度优先队列。

算法步骤：

初始化队列。

迭代终止条件

队列中只存储 行列索引即可。

vector<vector<int> > vis(m, vector<int>(n, 0));

记录单元格是否被访问。

```c++
class Solution {
public:
    //获取格子的位数和
    int get(int x){
        int res = 0;
        for(; x; x/=10){
            res+=x%10;
        }
        return res;

    }
    int movingCount(int m, int n, int k) {
        //只需要考虑向下和向右的方向
        int dx[2] = {0,1};
        int dy[2] = {1,0};
        if(k==0)return 1;
        queue<pair<int,int>>que;
        vector<vector<int>>res(m,vector<int>(n,0));
        que.push(make_pair(0,0));
        res[0][0] = 1;
        int ans = 1;
        while(!que.empty()){
            auto [x,y] = que.front();
            que.pop();
            for(int i = 0; i<2; i++){
                int tx = dx[i] + x;
                int ty = dy[i] + y;
                //判断超过边界的条件
                if(tx<0 || ty<0 || tx>=m || ty>=n
                || get(tx)+get(ty)>k||res[tx][ty]){
                    continue;
                }
                que.push(make_pair(tx,ty));
                res[tx][ty] = 1;//标记走过
                ans++;
            }

        }
        return ans;
       }
 };
```

方法二：深度优先和回溯

首先进行数位之和的计算

设一数字 x ，向下取整除法符号 / ，求余符号 % ，则有：

x%10 ：得到 x 的个位数字；
x/10 ： 令 x 的十进制数向右移动一位，即删除个位数字。

```c++
int get(int x){
        int res = 0;
        for(; x; x/=10){
            res+=x%10;
        }
        return res;

    }
```

其次：可达解分析

根据数位和增量公式分析，数位每逢进位突变一次。

所以矩阵中的**数位和解**满足等腰三角形，每个三角形的直角顶点位于 0, 10, 20, ...0,10,20,... 等数位和突变的矩阵索引处 。

三角形内的解虽然都满足数位和要求，但由于机器人每步只能走一个单元格，而三角形间不一定是连通的，因此机器人不一定能到达，称之为 不可达解 ；同理，可到达的解称为 可达解 （本题求此解） 。

![img](https://pic.leetcode-cn.com/1603026306-OdpwLi-Picture1.png)

![img](https://pic.leetcode-cn.com/1603026306-daxIuh-Picture4.png)

![Picture9.png](https://pic.leetcode-cn.com/1603024999-XMpudY-Picture9.png)

深度优先遍历需要注意的是**剪枝：** 在搜索中，遇到数位和超出目标值、此元素已访问，则应立即返回，称之为 `可行性剪枝` 。其步骤如下：

1、递归的参数：i,j 当前的行列索引，visited当前元素是否被访问 .返回值为到达的格子数。

2、递归终止条件：行列索引越界

​								数位和超出目标值

​								已经访问过该元素 则返回0 不计入格子数。

递归处理：标记单元格

​				搜索下一个单元格：向下或者向右。回溯返回值： 返回 1 + 右方搜索的可达解总数 + 下方搜索的可达解总数，代表从本单元格递归搜索的可达解总数。


```c++
class Solution {
public:
    int m_size,n_size,k_size;
    //获取格子的位数和
    int get(int x){
        int res = 0;
        for(; x; x/=10){
            res+=x%10;
        }
        return res;

    }
    int dfs(int i, int j, vector<vector<bool>>&visited){
        if(i>=m_size || j>=n_size || visited[i][j] ==true ||get(i)+get(j)>k_size )return 0;
        visited[i][j] = true;
        return 1+dfs(i+1,j,visited)+dfs(i,j+1,visited);
    }
      int movingCount(int m, int n, int k) {
        //深度优先搜索
        //剪枝 数位超过目标值 元素访问过 
        vector<vector<bool>>vis(m,vector<bool>(n,false));
        return dfs(0,0,vis);
        

    }
};
```

#### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

 为了让您更好地理解问题，以下面的二叉搜索树为例：

![img](https://assets.leetcode.com/uploads/2018/10/12/bstdlloriginalbst.png)

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

![img](https://assets.leetcode.com/uploads/2018/10/12/bstdllreturndll.png)

本题思路：二叉搜索树转换成一个“排序的双向链表”

1、节点从小到大排序，故使用中序遍历。

```c++
void dfs(Node *cur){
        if(cur == nullptr)return;
        dfs(cur->left);
        cout<<cur->val<<endl;
        dfs(cur->right);
    }
```

2、构建双向链表：要有前驱结点pre和当前节点cur。

有pre->right = cur, cur->left = pre.

3、循环链表：设置链表头head和尾结点tail相连接

head->left = tail, tail ->right = head.

![Picture1.png](https://pic.leetcode-cn.com/1599401091-PKIjds-Picture1.png)

dfs具体步骤如下

```c++
 void dfs(Node *cur){
        if(cur == nullptr)return;
        dfs(cur->left);
        //构建双向链表
        //pre 用于记录当前节点左侧的节点 即上一个节点
        if(pre!=nullptr)pre->right = cur;
        //pre为空时表示的是 当前为头结点
        else head = cur;
        cur->left = pre;
        //更新操作
        pre = cur;
      
```



```c++
class Solution {
public:
    Node *pre, *head;
    //对二叉搜索树进行中序遍历 pre 和 cur
    void dfs(Node *cur){
        if(cur == nullptr)return;
        dfs(cur->left);
        //处理为双向链表
        //pre 用于记录当前节点左侧的节点 即上一个节点
        if(pre!=nullptr)pre->right = cur;
        //pre为空时表示的是 当前为头结点
        else head = cur;
        cur->left = pre;
        //更新操作
        pre = cur;
        dfs(cur->right);
    }
    Node* treeToDoublyList(Node* root) {
        if(root == nullptr)return nullptr;
        dfs(root);
        //初始化为循环链表
        pre->right = head;
        head->left = pre;//进行首尾节点的相互指向

       
        return head;
        
    }
};
```

#### [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]
```

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

```
输入：root = [1,2,3], targetSum = 5
输出：[]
```

- 树中节点总数在范围 `[0, 5000]` 内

- `-1000 <= Node.val <= 1000`

- `-1000 <= targetSum <= 1000`

  1、采用深度优先搜素遍历

  1、采用前序遍历，遍历树的所有节点

  2、记录根节点到当前节点的路径，当满足路径中各节点值的和等于target加入结果列表。

  ![Picture1.png](https://pic.leetcode-cn.com/697ce69b1c2df33091587432fd86a4f51559c9a26afa79c415a963e3ec42c99d-Picture1.png)

  深度搜素步骤：
  1、递归参数：cur当前节点，target目标值

  2、终止条件：到叶子节点终止 root==NULL时终止、

  3、递归工作

     1、将路径加入到path，更新当前target

  ​	2、判断路径是否满足，满足加入result,否则继续找、。

  ​	3、前序遍历

  ​	4、回溯路径 path.pop_back();

```c++
class Solution {
public:
    vector<int>path;
    vector<vector<int>>result;
    void dfs(TreeNode* cur, int target){
        if(cur == NULL)return;
        path.emplace_back(cur->val);
        target-=cur->val;
        //逻辑处理部分 判断什么时候需要将路径加入到结果
        if(target==0 && cur->left ==nullptr && cur->right == nullptr){
            result.emplace_back(path);
        }
        dfs(cur->left,target);
        dfs(cur->right,target);
        path.pop_back();
        return;
    }
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        dfs(root,target);
        return result;

    }
};
```

#### [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

给定一棵二叉搜索树，请找出其中第 `k` 大的节点的值。

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4

```

#### [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

给定一棵二叉搜索树，请找出其中第 `k` 大的节点的值。

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
```

- 1 ≤ k ≤ 二叉搜索树元素个数

  本题比较简单。

  1、首先用result存储中序遍历，结果为从小到大排列。

  2、只需输出第result.size()-k个元素即可。

  ![Picture1.png](https://pic.leetcode-cn.com/4ebcaefd4ecec0d76bfab98474dfed323fb86bfcd685d1a5bf610200fdca4405-Picture1.png)

  ```c++
  class Solution {
  public:
      vector<int>result;
      void dfs(TreeNode *root){
          if(root == NULL)return;
          dfs(root->left);
          result.emplace_back(root->val);
          dfs(root->right);
      }
      int kthLargest(TreeNode* root, int k) {
          dfs(root);
          int kVal = result.size()-k;
          return result[kVal];
          
      }
  };
  ```

  左根右--> 右根左 （因为是K大，右根左是递减排序）这是参考别人的。

  迭代法

```c++
class Solution {
public:
    int kthLargest(TreeNode* root, int k) {
        stack<TreeNode*> stk;
        
        while(root || !stk.empty()){
            while(root){
                stk.push(root);
                root = root->right;
            }
            root = stk.top();
            stk.pop();
            if(--k == 0)
                return root->val;
            
            root = root->left;
        }
        return -1;
    }
};
```

递归法

```cpp
class Solution {
public:
    int kthLargest(TreeNode* root, int k) {
        DFS(root, k);
        return ans;
    }
    void DFS(TreeNode* node, int& k) {
        if (node == nullptr) return;
        DFS(node -> right, k);
        if (--k == 0) {
            ans = node -> val;
            return;
        }   
        DFS(node -> left, k);
    }
private:
    int ans;
};
```



#### [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 [3,9,20,null,null,15,7]，

```
  3
   / \
  9  20
    /  \
   15   7
   返回它的最大深度 3 。
   节点总数 <= 10000
```

![Picture1.png](https://pic.leetcode-cn.com/9b063f1f2b7ba125b97a2a11c5f774c0f8ff4df594696993a8eb8282750dae0d-Picture1.png)

方法一：深度优先搜索

1、返回条件：当root为空，返回0；

2、递归计算左子树深度，右子树深度

3、树的深度为：左右子树深度最大值+1；

```c++
int maxDepth(TreeNode* root) {
        if(root == NULL)return 0;
        return max(maxDepth(root->left),maxDepth(root->right))+1;
```

方法2：广度优先（层序遍历）

利用队列进行辅助实现，每遍历一层加1；

```c++
if(root == NULL)return 0;
    queue<TreeNode*>que;
    int res = 0;
    que.push(root);
    while(!que.empty()){
        int size = que.size();
        for(int i =0; i<size; i++){
            TreeNode *node = que.front();
            que.pop();
            if(node->left)que.push(node->left);
            if(node->right)que.push(node->right);

        }
        res++;
    }
    return res;
```

#### [404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/)

计算给定二叉树的所有左叶子之和。

```
  3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
```

首先统计的是左叶子节点，首先该节点是左子节点，其次要是叶子节点。

我们需要对整棵树进行遍历，当遍历到node节点时，它的左子节点是叶子节点，那就累加入答案。遍历整棵树的方法有深度优先搜索和广度优先搜索，下面分别给出了实现代码。

方法一：深度优先（前序遍历）

```c++
class Solution {
public:
    bool isLeafNode(TreeNode *root){
        return !root->left && !root->right;
    }
    int dfs(TreeNode *node){
        int sum = 0;
        if(node == NULL)return 0;
        //左子树的左叶子节点值
        if(node->left){
            sum+= isLeafNode(node->left) ?node->left->val : dfs(node->left);
        }
        //右子树的左叶子节点
        if(node->right && !isLeafNode(node->right)){
            sum+= dfs(node->right);
        }
        return sum;
    }
    int sumOfLeftLeaves(TreeNode* root) {
        if(root == NULL)return 0;
        return dfs(root);
    }
};
```

方法2 宽度优先

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool isLeafNode(TreeNode *root){
        return !root->left && !root->right;
    }
    int sumOfLeftLeaves(TreeNode* root) {
        if(root == NULL)return 0;
        queue<TreeNode*>que;
        int ans = 0;
        que.emplace(root);
        while(!que.empty()){
            TreeNode * node = que.front();
            que.pop();
            if(node->left){
                if(isLeafNode(node->left)){
                    ans+= node->left->val;

                }else que.emplace(node->left);
            }
            if(node->right){
                if(!isLeafNode(node->right)){
                    que.emplace(node->right);
                }
            }
        }
        return ans;
       
    }
};
```

#### [513. 找树左下角的值](https://leetcode-cn.com/problems/find-bottom-left-tree-value/)

给定一个二叉树的 **根节点** `root`，请找出该二叉树的 **最底层 最左边** 节点的值。

假设二叉树中至少有一个节点。

![img](https://assets.leetcode.com/uploads/2020/12/14/tree1.jpg)

```
输入: root = [2,1,3]
输出: 1
```

![img](https://assets.leetcode.com/uploads/2020/12/14/tree2.jpg)

```
输入: [1,2,3,4,null,5,6,null,null,7]
输出: 7
```

本题求得是二叉树左下角的值：也就是首先要求是最底层 其次是最左边。

不要求一定是左子节点。

方法一：深度优先遍历

```c++
class Solution {
public:
    int maxLen = INT_MIN; //最大深度
    int maxLeftValue; //最大深度对于最左节点值
    void dfs(TreeNode *root,int leftLen){
        //遇到叶子节点返回
        if(root->left == NULL && root->right ==NULL){
            if(leftLen > maxLen){
                maxLen = leftLen;
                maxLeftValue = root->val;
            }
            return;
        }

        if(root->left)dfs(root->left,leftLen+1);
        if(root->right)dfs(root->right,leftLen+1);
    }

    int findBottomLeftValue(TreeNode* root) {
        dfs(root,0);
        return maxLeftValue;

    }
};
```

方法二：广度优先遍历

```c++
class Solution {
public:
    int maxLen = INT_MIN; //最大深度
    int maxLeftValue; //最大深度对于最左节点值
    void dfs(TreeNode *root,int leftLen){
        //遇到叶子节点返回
        if(root->left == NULL && root->right ==NULL){
            if(leftLen > maxLen){
                maxLen = leftLen;
                maxLeftValue = root->val;
            }
            return;
        }

        if(root->left)dfs(root->left,leftLen+1);
        if(root->right)dfs(root->right,leftLen+1);
        
    }

    int findBottomLeftValue(TreeNode* root) {
        // dfs(root,0);
        // return maxLeftValue;

        //广度优先遍历
        queue<TreeNode*>que;
        que.push(root);
        int res = 0;
        while(!que.empty()){
            int size = que.size();
            for(int i =0; i<size; i++){
                TreeNode *node = que.front();
                que.pop();
                if(i == 0)res = node->val;
                if(node->left)que.push(node->left);
                if(node->right)que.push(node->right);

            }
        }
        return res;

    }
};
```



#### [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

```
给定二叉树 [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
返回 true 。
给定二叉树 [1,2,2,3,3,null,null,4,4]
 	   1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。
0 <= 树的结点个数 <= 10000

```

本题采用深度优先遍历：
可以采用前序遍历 或者后序遍历



```c++
class Solution {
public:
    //利用后序遍历迭代 判断每一个子树是否满足平衡二叉树
    int height(TreeNode* node){
        if(node == NULL)return 0;
        int left = height(node->left);
        int right = height(node->right);
        if(left == -1|| right ==-1 || abs(left-right)>1)return -1;
        else return max(left,right)+1;
    }
    bool isBalanced(TreeNode* root) {
        return height(root)>=0;
        

    }
};
```

#### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
```

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

```
输入：root = [1,2,3], targetSum = 5
输出：false
输入：root = [], targetSum = 0
输出：false
```

方法1 dfs



方法2 dfs+回溯

这里的回溯指 利用 DFS 找出从根节点到叶子节点的所有路径，只要有任意一条路径的 和 等于 sum，就返回 True。

```cpp
class Solution {
private:
    bool backtracking(TreeNode *cur,int sum){
        // sum-=cur->val;
        if(cur->left == nullptr && cur->right == nullptr && sum==0)return true;
        if(!cur->left &&!cur->right)return false;

        if(cur->left){
            if(backtracking(cur->left,sum-cur->left->val))return true;
        }
        if(cur->right){
            if(backtracking(cur->right,sum-cur->right->val))return true;
        }
        return false;
    }
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if(root == nullptr)return false;
        
        return backtracking(root,targetSum-root->val);
    }
};
```

#### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg)

输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]

![img](https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg)

```
输入：root = [1,2,3], targetSum = 5
输出：[]
```

dfs+回溯

```c++
class Solution {
private:
    vector<vector<int>>res;
    vector<int>path;
    void dfs(TreeNode *cur,int sum){
        if(!cur->left &&!cur->right && sum==0){
            res.emplace_back(path);
            return;
        }
        
        if(!cur->left &&!cur->right)return;
        
        if(cur->left){
            path.emplace_back(cur->left->val);
            dfs(cur->left,sum-cur->left->val);
            path.pop_back();
        }
        if(cur->right){
            path.emplace_back(cur->right->val);
            dfs(cur->right,sum-cur->right->val);
            path.pop_back();
        }
        
    }
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if(root == nullptr)return res;
        path.push_back(root->val);
        dfs(root,targetSum-root->val );
        return res;
    }
};
```

#### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

![img](https://assets.leetcode.com/uploads/2021/04/09/pathsum3-1-tree.jpg)



```
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：3
```

方法1 双重递归

利用p为节点，往下搜索。但是这种方法有很多重复计算O（n2）时间复杂度

```c++
class Solution {
public:
    int rootSum(TreeNode *root, int targetSum){
        if(root == nullptr)return 0;

        int ret = 0;
        if(root->val == targetSum)ret++;
        ret += rootSum(root->left, targetSum - root->val);
        ret += rootSum(root->right,targetSum -root->val);
        return ret;
    }
    
    int pathSum(TreeNode* root, int targetSum) {
        if(root == nullptr)return 0;

        int ret = rootSum( root, targetSum);
        ret += pathSum(root->left,targetSum);
        ret += pathSum(root->right,targetSum);
        return ret;

    }
};
```

方法2 前缀和

由根结点到当前结点的路径上所有节点的和。我们利用先序遍历二叉树，记录下根节点root 到当前节点 p 的路径上除当前节点以外所有节点的前缀和，在已保存的路径前缀和中查找是否存在前缀和刚好等于当前节点到根节点的前缀和 curr 减去 targetSum。

```c++
class Solution {
private:
    unordered_map<long long,int>prefix;
    int dfs(TreeNode *root, long long sum ,int target){
        if(root == nullptr)return 0;
        int res = 0;
        sum+=root->val;
        if(prefix.count(sum-target)){
            //满足前缀和之差为target
            res = prefix[sum-target];
        }
        prefix[sum]++;
        res += dfs(root->left,sum,target);
        res+=dfs(root->right, sum, target);
        prefix[sum]--;
        return res;
    }
public:
    int pathSum(TreeNode* root, int targetSum) {
        if(root == nullptr)return 0;
        prefix[0]=1;
        return dfs(root,0,targetSum);
    }
};
```

#### [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

![img](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

![img](https://assets.leetcode.com/uploads/2020/11/04/word-1.jpg)

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
输出：true

```c++
class Solution {
public:
    int dx[4] = {-1,0,1,0},dy[4] = {0,1,0,-1};
    bool dfs(vector<vector<char>>& board,string &word,int u, int x, int y){
        if(x<0 || x>=board.size() ||y<0 || y>=board[0].size()||word[u]!=board[x][y])return false;
        if(u == word.size()-1){
            return true;
        }
        
        char t = board[x][y];
        board[x][y] = '.';
        for(int i = 0; i<4; i++){
            int tx = dx[i]+x, ty = dy[i]+y;
            // if(tx<0 || tx>=board.size() ||ty<0 || ty>=board[0].size() )continue;
            if(dfs(board,word,u+1,tx,ty))return true;
        }
        board[x][y] = t;
        return false;
        
    }
    bool exist(vector<vector<char>>& board, string word) {
        for(int i = 0; i<board.size(); i++){
            for(int j = 0; j<board[i].size(); j++){
                if(dfs(board,word,0,i,j))return true;
            }
        }
        return false;
    }
};
```



# 分治法

#### [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。

假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

![img](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Input: preorder = [-1], inorder = [-1]
Output: [-1]

```

```c++
class Solution {
private:
    vector<int>preorder;
    unordered_map<int,int>hashmap;
    TreeNode *recur(int root, int left, int right){
        if(left > right)return nullptr;
        int i = hashmap[preorder[root]];
        TreeNode *node = new TreeNode(preorder[root]);
        node->left = recur(root+1,left,i-1);
        node->right = recur(root+1+i-left,i+1,right);
        return node;
    }
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        this->preorder = preorder;
        for(int i = 0; i<inorder.size(); i++){
            hashmap[inorder[i]] = i;
        }
        return recur(0,0,inorder.size()-1);
    }
};
```



#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

给定两个整数数组 inorder 和 postorder ，其中 inorder 是二叉树的中序遍历， postorder 是同一棵树的后序遍历，请你构造并返回这颗 二叉树 。

 ![img](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

```
输入：inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
输出：[3,9,20,null,null,15,7]
输入：inorder = [-1], postorder = [-1]
输出：[-1]
1 <= inorder.length <= 3000
postorder.length == inorder.length
-3000 <= inorder[i], postorder[i] <= 3000
inorder 和 postorder 都由 不同 的值组成
postorder 中每一个值都在 inorder 中
inorder 保证是树的中序遍历
postorder 保证是树的后序遍历

```

```c++
class Solution {
private:
    vector<int>postorder;
    unordered_map<int,int>hashmap;
    TreeNode* recur(int root,int left, int right){
        if(left>right)return nullptr;
        TreeNode *node = new TreeNode(postorder[root]);
        int i = hashmap[postorder[root]];
        node->left = recur(root-1-right+i,left,i-1);
        node->right = recur(root-1,i+1,right);
        return node;
    }
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        this->postorder = postorder;
        for(int i = 0; i<inorder.size(); i++)hashmap[inorder[i]] = i;
        return recur(inorder.size()-1,0,inorder.size()-1);
    }
};
```



#### [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：

```
 	 5
    / \
   2   6
  / \
 1   3
输入: [1,6,3,2,5]
输出: false
输入: [1,3,2,6,5]
输出: true
数组长度 <= 1000
```

后序遍历定义： [ 左子树 | 右子树 | 根节点 ] ，即遍历顺序为 “左、右、根” 。
二叉搜索树定义： 左子树中所有节点的值 < 根节点的值；右子树中所有节点的值 > 根节点的值；其左、右子树也分别为二叉搜索树。



![Picture1.png](https://pic.leetcode-cn.com/4a2780853b72a0553194773ff65c8c81ddcc4ee5d818cb3528d5f8dd5fa3b6d8-Picture1.png)

判断每一个子树的性质，判断其是否满足二叉搜索数。

终止条件： 当 i \geq ji≥j ，说明此子树节点数量 \leq 1≤1 ，无需判别正确性，因此直接返回 truetrue ；
递推工作：
划分左右子树： 遍历后序遍历的[i,j] 区间元素，寻找 第一个大于根节点 的节点，索引记为 m 。此时，可划分出左子树区间 [i,m−1] 、右子树区间[m,j−1] 、根节点索引 j 。
判断是否为二叉搜索树：
左子树区间[i,m−1] 内的所有节点都应 <postorder[j] 。而第 1.划分左右子树 步骤已经保证左子树区间的正确性，因此只需要判断右子树区间即可。
右子树区间 [m,j−1] 内的所有节点都应 >postorder[j] 。实现方式为遍历，当遇到 ≤postorder[j] 的节点则跳出；则可通过 p=j 判断是否为二叉搜索树。

```c++
class Solution {
public:
    //后序 左右 根
    //从左到右第一个大于根节点的值索引为m可以划分左右子树
    bool recur(vector<int>& postorder,int left,int right){
        if(left>=right)return true;
        int i = left;
        while(postorder[i]<postorder[right])i++;
        int nowIndex = i;
        while(postorder[i]>postorder[right])i++;
        return i==right && recur(postorder,left,nowIndex-1) && recur(postorder,nowIndex,right-1);

    }
    bool verifyPostorder(vector<int>& postorder) {
        return recur(postorder,0,postorder.size()-1);

    }
};
```

#### [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

```
输入：x = 2.00000, n = 10
输出：1024.00000
输入：x = 2.10000, n = 3
输出：9.26100
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
-100.0 < x < 100.0
-231 <= n <= 231-1
-104 <= xn <= 104
```

快速幂是二分思想的应用。

分别为n为奇数和偶数两种情况讨论
$$
x^n = x^{n/2} \times x^{n/2} = (x^2)^{n/2}
$$
当 n 为偶数： 
$$
x^n = (x^2)^{n//2}
$$
当 n 为奇数：
$$
 x^n = x(x^2)^{n//2}x 
$$
即会多出一项 x ；



```c++
class Solution {
public:
    double myPow(double x, int n) {
        double res = 1.0;
        //本题c++采用int n会溢出
        long a = n;
        if(a==0&&x==0.0)return 0;
        if(a<0){
            x=1/x;
            a = -a;
        }
        while(a){
            if(a&1==1)res*=x;
            x*=x;
            a>>=1;
        }
        return res;
        

    }
};
```

# 贪心算法

#### [5236. 美化数组的最少删除数](https://leetcode-cn.com/problems/minimum-deletions-to-make-array-beautiful/)

给你一个下标从 0 开始的整数数组 nums ，如果满足下述条件，则认为数组 nums 是一个 美丽数组 ：

nums.length 为偶数
对所有满足 i % 2 == 0 的下标 i ，nums[i] != nums[i + 1] 均成立
注意，空数组同样认为是美丽数组。

你可以从 nums 中删除任意数量的元素。当你删除一个元素时，被删除元素右侧的所有元素将会向左移动一个单位以填补空缺，而左侧的元素将会保持 不变 。

返回使 nums 变为美丽数组所需删除的 最少 元素数目。

```
输入：nums = [1,1,2,3,5]
输出：1
解释：可以删除 nums[0] 或 nums[1] ，这样得到的 nums = [1,2,3,5] 是一个美丽数组。可以证明，要想使 nums 变为美丽数组，至少需要删除 1 个元素。

输入：nums = [1,1,2,2,3,3]
输出：2
解释：可以删除 nums[0] 和 nums[5] ，这样得到的 nums = [1,2,2,3] 是一个美丽数组。可以证明，要想使 nums 变为美丽数组，至少需要删除 2 个元素。

1 <= nums.length <= 105
0 <= nums[i] <= 105
```

方法1 利用栈的思想来求res

```c++
class Solution {
public:
    int minDeletion(vector<int>& nums) {
        //利用栈的思想来模拟
        int last = nums[0];
        int flag = 1, res = 0;
        for(int i = 1; i< nums.size(); i++){
            if(flag){
                //前一个数为栈顶 栈顶为偶数下标 需要判断
                if(nums[i] == last){
                    res++;
                    //这个时候就删除nums[i]
                }else flag = 0;
            }
            else{
                //栈顶为奇数下标 随便入
                flag = 1;
                //更新last
                last = nums[i];
            }
        }
        if((nums.size()-res)%2)return res+1;
        return res;
    }
};
```

方法2 贪心算法

```c++
class Solution {
public:
    int minDeletion(vector<int>& nums) {
        //贪心算法 当前数可以作为数对中的第二个数则保留 
        int n = nums.size();
        int ans = 0;
        for(int i = 0; i<n-1; i++){
            if(nums[i] == nums[i+1])ans++;
            else i++;
        }
        if((n-ans)%2)return ans+1;
        return ans;
        
    }
};
```



# 动态规划

## 编辑距离总结篇

### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
输入：nums = [0,1,0,3,2,3]
输出：4
输入：nums = [7,7,7,7,7,7,7]
输出：1
1 <= nums.length <= 2500
-104 <= nums[i] <= 104
```

本题主要思路是采用动态规划，采用动态规划五步法。(序列不要求连续)

1、dp[i]的定义：位置i的最长升序子序列是 从0到i-1位置的最长升序子序列+1的最大值。

2、求状态转移方程

if（nums[i]>nums[j]）dp[i]=max(dp[i],dp[j]+1)  j为小于i的数

3、dp[i]的初始化

起始大小为1；

4、遍历顺序

从左往右0-i

j为 0~i-1；

5、举例验证

输入：[0,1,0,3,2]，dp数组的变化如下：

![300.最长上升子序列](https://img-blog.csdnimg.cn/20210110170945618.jpg)

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int lenth = 1;
        vector<int>dp(nums.size(),1);
        for(int i = 1; i<nums.size(); i++){
            for(int j = 0; j<i; j++){
               if(nums[i]>nums[j]) dp[i] = max(dp[i],dp[j]+1);
            }
            if(dp[i]>lenth)lenth = dp[i];
        }
        return lenth;

    }
};
```

针对最长递增子序列做少许更改，返回最长严格递增子序列。

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：[2,3,7,101]
解释：最长递增子序列是 [2,3,7,101]。
```

```java
class Solution {
    public List<Integer> lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        List<List<Integer>> paths = new ArrayList<>();
        int max = 0;
        int index = -1;
        for(int i = 0;i<n;i++){
            dp[i] = 1;
            List<Integer> temp1 = new ArrayList<>();
            temp1.add(nums[i]);
            paths.add(temp1);
            for(int j = 0;j<i;j++){
                if(nums[i]>nums[j]){
//                    dp[i] = Math.max(dp[i],dp[j]+1);
                    if(dp[i]<dp[j]+1){
                        List<Integer> temp = new ArrayList<>(paths.get(j));
                        temp.add(nums[i]);
                        paths.remove(i);
                        paths.add(temp);
                        dp[i]=dp[j]+1;
                    }
                }
            }
            if(max<dp[i]){
                max = dp[i];
                index = i;
            }
        }
        if(index ==-1) return null;
        return paths.get(index);
    }
}
```



### [674. 最长连续递增序列](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)

给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

```
输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 
输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。
```

本题与上一道题相比，限制为了连续递增子序列。

那么可以采用动态规划或者贪心求解本题

方法1：贪心算法

对于下标范围 [l,r][l,r] 的连续子序列，如果对任意 l≤i<r 都满足 nums[i]<nums[i+1]，则该连续子序列是递增序列。

如果nums[l-1]<nums[l],可以将nums[l-1]加入到递增子序列

nums[r]<nums[r+1]，可以将nums[r+1]加入到子序列。

```c++
 //贪心
        if(nums.size()==0)return 0;
        int result = 1;
        int count = 1;
        for(int i = 1; i<nums.size(); i++){
            if(nums[i]>nums[i-1])count++;
            else count = 1;
            result = max(result,count);
        }
        return result;
```

方法2：动态规划

```c++
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        //本题要求的子序列为连续的子序列
        //动态规划
        if(nums.size()==0)return 0;
        vector<int>dp(nums.size(),1);
        int res = 1;
        for(int i = 0; i<nums.size()-1; i++){
            if(nums[i]<nums[i+1])dp[i+1] = dp[i]+1;
            //res = max(dp[i+1],res);
            if(res<dp[i+1])res = dp[i+1];
        }
        return res;
        }
}；
```

### [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。


```

方法1 哈希表+滑动窗口

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int>hash;
        int res = 0;
        for(int i = 0,j=0; i<s.size(); i++){
            hash[s[i]]++;
            while(j<i && hash[s[i]]>1)hash[s[j++]]--;
            res = max(res,i-j+1);
        }
        return res;
    }
};
```

方法2 哈希表+动态规划（空间压缩了）

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int>hash;
        // tmp为动态规划中的dp数组
        int res = 0, tmp = 0;
        for(int j = 0; j<s.size(); j++){
            int i = -1;
            if(hash.count(s[j])){
                i = hash[s[j]];//找到重复字符第一次出现的位置
            }
            hash[s[j]]= j;
            // hash.push_back(s[j],j);
            tmp = tmp<j-i ?tmp+1:j-i;
            res = max(res,tmp);
        }
        return res;
    }
};
```



### [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

给两个整数数组 `A` 和 `B` ，返回两个数组中公共的、长度最长的子数组的长度。

```
输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：
长度最长的公共子数组是 [3, 2, 1] 。
1 <= len(A), len(B) <= 1000
0 <= A[i], B[i] < 100
```

本题要求找到数组的交集。

注意题目中说的子数组，其实就是连续子序列。这种问题动规最拿手，动规五部曲分析如下：

1、确定dp数组（dp table）以及下标的含义:dp[i] [j]:以下标i-1为结尾的a,和以下标j为结尾的b数组，最长的重复子数组长度。

2、递推公式

a[i-1] ==b[i-1]时候，dp[i] [j]=dp[i-1] [j-1]+1;

3、dp数组如何初始化

根据dp[i][j]的定义，dp[i] [0] 和dp[0] [j]其实都是没有意义的！

但dp[i][0] 和dp[0][j]要初始值，因为 为了方便递归公式dp[i] [j] = dp[i - 1] [j - 1] + 1;

所以dp[i] [0] 和dp[0] [j]初始化为0。

4、确定遍历顺序

外层for循环遍历A，内层for循环遍历B。

```c++
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(),n = nums2.size();
        vector<vector<int>>dp(m+1,vector<int>(n+1,0));
        int res=0;
        for(int i = 1; i<=m; i++){
            for(int j =1 ; j<=n; j++){
                if(nums2[j-1]==nums1[i-1])dp[i][j] = dp[i-1][j-1]+1;
                // res = max(res,dp[i][j]);
                if(dp[i][j]>res)res = dp[i][j];
            }
            
        }
        return res;

    }
};
```



### [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0 。
1 <= text1.length, text2.length <= 1000
text1 和 text2 仅由小写英文字符组成。
```

和最长重复数组相比，本题不需要公共序列连续了，保持相对顺序即可。

下面将采用动态规划法

1、dp[i] [j]数组的下标及其含义：

长度为[0,i-1]的字符串a和长度为[0,j-1]的字符串b的最长公共字符列长度。

2、递推公式

a[i-1]==b[i-1]时 dp[i] [j] = dp[i-1] [j-1]+1；

a[i-1] != b[i-1] dp[i] [j] = max(dp[i-1] [j],dp[i] [j-1])

3、dp数组如何初始化

a[0, i-1]和空串的最长公共子序列自然是0，所以dp[i] [0] = 0;

同理dp[0] [j] = 0;

4、遍历顺序

![1143.最长公共子序列](https://img-blog.csdnimg.cn/20210204115139616.jpg)



```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n= text2.size();
        vector<vector<int>>dp(m+1,vector<int>(n+1,0));
        int res = 0;
        for(int i = 1; i<=m; i++){
            for(int j = 1; j<=n; j++){
                if(text1[i-1]==text2[j-1])dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
                res = max(res,dp[i][j]);

            }
        }
        return res;

    }
};
```



### [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。

现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足满足：

 nums1[i] == nums2[j]
且绘制的直线不与任何其他连线（非水平线）相交。
请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。

以这种方法绘制线条，并返回可以绘制的最大连线数。

![img](https://assets.leetcode.com/uploads/2019/04/26/142.png)

```
输入：nums1 = [1,4,2], nums2 = [1,2,4]
输出：2
解释：可以画出两条不交叉的线，如上图所示。 
但无法画出第三条不相交的直线，因为从 nums1[1]=4 到 nums2[2]=4 的直线将与从 nums1[2]=2 到 nums2[1]=2 的直线相交。

输入：nums1 = [2,5,1,2,5], nums2 = [10,5,2,1,5,2]
输出：3
输入：nums1 = [1,3,7,1,7,5], nums2 = [1,9,2,5,1]
输出：2
1 <= nums1.length, nums2.length <= 500
1 <= nums1[i], nums2[j] <= 2000
```

本题求解和最长公共子序列基本一致的。

因为本题绘制一些连接两个数字 A[i] 和 B[j] 的直线，只要 A[i] == B[j]，且直线不能相交！

直线不能相交，这就是说明在字符串A中 找到一个与字符串B相同的子序列，且这个子序列不能改变相对顺序，只要相对顺序不改变，链接相同数字的直线就不会相交。

那么其实不相交的线又回到了求公共子序列上来了。

```c++
class Solution {
public:
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(),n = nums2.size();
        vector<vector<int>>dp(m+1,vector<int>(n+1,0));
        int res = 0;
        for(int i =1; i<=m; i++){
            for(int j= 1; j<=n; j++){
                if(nums1[i-1] == nums2[j-1])dp[i][j] = dp[i-1][j-1]+1;
                else dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
                res = max(res,dp[i][j]);
            }
        }
        return res;

    }
};
```



### [53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组** 是数组中的一个连续部分。

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
输入：nums = [1]
输出：1
输入：nums = [5,4,-1,7,8]
输出：23
```

这类题目中限定了是连续子序列的问题，通常还可以考虑贪心算法求解。

方法一：贪心算法

局部最优的情况下记录最大的“连续和”，从而可以推导出全局最优。

即从左到右遍历数组，如果前一个区间和为正数，加上nums[i]后为负数了，那么重新从i+1区间开始累积。

```c++
/1贪心算法
        // int result = INT32_MIN;
        // int count = 0;
        // for(int i = 0; i<nums.size(); i++){
        //     count += nums[i];
        //     if(count > result )result = count;
        //     if(count<=0)count = 0;
        // }
        // return result;
```



方法2 动态规划

依旧是动态规划五部法

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        //2动态规划
        if(nums.size() == 0)return 0;
        vector<int>dp(nums.size(),0);
        dp[0] = nums[0];
        int result = dp[0];
        for(int i = 1; i<nums.size(); i++){
            dp[i] = max(dp[i-1]+nums[i],nums[i]);
            if(dp[i] > result)result = dp[i];
            // result = max(result,dp[i]);
        }
        return result;

    }
};
```



### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
输入：s = "cbbd"
输出："bb"
输入：s = "a"
输出："a"
1 <= s.length <= 1000
```

方法一：中心扩展算法

![image-20220110203551849](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220110203551849.png)

所有回文串都是由最上面两种情况开始扩散的，即i=j必定是回文，si = si+1时，情况2也满足回文。其他情况可以通过状态转移到这两种情况。

通过枚举所有的回文中心，实现对字串的扩展，判断是否属于回文串。直到无法扩展为止，此时的回文串长度即为此「回文中心」下的最长回文串长度。我们对所有的长度求出最大值，即可得到最终的答案。

```c++
class Solution {
public:
    //方法一 中心扩展算法
    pair<int,int>expandCenter(const string &s, int left,int right){
        while(left>=0 && right<s.size() && s[left] == s[right]){
            --left;
            ++right;
        }
        return {left+1,right-1};
    }
    string longestPalindrome(string s) {
        int start = 0, end = 0;
        for(int i =0; i<s.size(); i++){
            //中心为一个元素和两个元素的情况
            auto [left1,right1] = expandCenter(s,i,i);
            auto [left2,right2] = expandCenter(s,i,i+1);
            //更新start 和 end
            if(right1 - left1>end-start){
                start = left1;
                end = right1;
            }
            if(right2 - left2>end-start){
                start = left2;
                end = right2;
            }
        }

        return s.substr(start,end-start+1);


    }
};
```





### [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。

回文字符串 是正着读和倒过来读一样的字符串。

子字符串 是字符串中的由连续字符组成的一个序列。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

```
输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"
输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
1 <= s.length <= 1000
s 由小写英文字母组成
```

本题和53不同的地方是，本题需要枚举所有回文串，得到回文子串的个数。

那么这道题用动态规划求解是比较好的。

1、确定dp数组以及下标含义。

dp[i] [j]:区间范围[i,j]的字串是否是回文，是为true，不是则为false;

2、递推公式

判断s[i]和s[j]是否相等。

- [ ] s[i]和s[j]不相等，那么dp[i]] [j]为false。

- [ ] s[i]和s[j]相等,下面有三种情况推导

  下标i，j相等 肯定是回文，

  i和j相差1，那么也满足回文

  i和j相差大于1，那么缩小区间继续判断dp[i+1] [j-1]是否为true；

  

```c++
 int countSubstrings(string s) {
        //动态规划
        int res = 0;
        vector<vector<bool>>dp(s.size(),vector<bool>(s.size(),false));
        for(int i = s.size()-1; i>=0; i--)
            for(int j = i; j<s.size(); j++)
                if(s[i] == s[j]){
                    if(j-i<=1){
                        dp[i][j]=true;res++;

                    }
                    else if(dp[i+1][j-1]){
                       
                        res++;
                        dp[i][j]=true;
                    
                    }
                }
        return res;
       }
```

方法2 中心扩展法（双指针）

![image-20220110203551849](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220110203551849.png)

 上一题的最长回文子串稍微改动就可以解本题。

```c++
class Solution {
public:
    int countSubstrings(string s) {  
        //回文字串这道题更适合用双指针
        //以一个字符为中心或者两个字符为中心
         int res = 0;
         for(int i = 0; i<s.size();i++){
             res+=extend(s,i,i);
             res+=extend(s,i,i+1);
         }
         return res;


    }
    int extend(const string &s,int i , int j){
        int res = 0;
        while(i>=0 && j<s.size() && s[i]==s[j]){
            i--;
            j++;
            res++;
        }
        return res;
    }
};
```



### [516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)

给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

```
输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb" 。
输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb" 。
```



本题采用动态规划求解：

回文子序列不要求连续，前两题中的回文子串是要求连续的。

相比回文子串情况少了一些。

1.确定动态数组以及下标含义。

dp[i] [j]:字符串s在[i,j]范围内的最长回文子序列。

2、递推公式

分两种情况：

s[i] = s[j]:

```c++
if(s[i]==s[j]){
                    dp[i][j] = dp[i+1][j-1]+2;
                }
```

![516.最长回文子序列](https://img-blog.csdnimg.cn/20210127151350563.jpg)

如果s[i]与s[j]不相同:

加入s[j]的回文子序列长度为dp[i + 1] [j]。

加入s[i]的回文子序列长度为dp[i] [j - 1]。

```c++
 dp[i][j] = max(dp[i+1][j],dp[i][j-1]);
```

3、初始化

对于i j 相等的情况需要单独初始化，因为递推公式中计算不到 i j相等的情况。

```c++
vector<vector<int>>dp(s.size(),vector<int>(s.size(),0));
        for(int i =0; i<s.size(); i++){
            dp[i][i] = 1;
        }
```

4、遍历顺序

显然从递推公式看出，i是由i+1推导的，所以i一定要从下到上遍历。

![516.最长回文子序列2](https://img-blog.csdnimg.cn/20210127151452993.jpg)

```c++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        vector<vector<int>>dp(s.size(),vector<int>(s.size(),0));
        for(int i =0; i<s.size(); i++){
            dp[i][i] = 1;
        }
        for(int i = s.size()-1; i>=0; i--){
            for(int j = i+1; j<s.size(); j++){
                if(s[i]==s[j]){
                    dp[i][j] = dp[i+1][j-1]+2;
                }else{
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }
        return dp[0][s.size()-1];

    }
};
```

# 并查集

#### [399. 除法求值](https://leetcode-cn.com/problems/evaluate-division/)

给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

```
输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
解释：
条件：a / b = 2.0, b / c = 3.0
问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
结果：[6.0, 0.5, -1.0, 1.0, -1.0 ]

输入：equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
输出：[3.75000,0.40000,5.00000,0.20000]

输入：equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
输出：[0.50000,2.00000,-1.00000,-1.00000]

1 <= equations.length <= 20
equations[i].length == 2
1 <= Ai.length, Bi.length <= 5
values.length == equations.length
0.0 < values[i] <= 20.0
1 <= queries.length <= 20
queries[i].length == 2
1 <= Cj.length, Dj.length <= 5
Ai, Bi, Cj, Dj 由小写英文字母与数字组成

```

本题考察并查集的思想

这道题是在「力扣」第 990 题（等式方程的可满足性）的基础上，在变量和变量之间有了倍数关系。由于 变量之间的倍数关系具有传递性，处理有传递性关系的问题，可以使用「并查集」，我们需要在并查集的「合并」与「查询」操作中 维护这些变量之间的倍数关系。

构建有向图

题目给出的 equations 和 values 可以表示成一个图，equations 中出现的变量就是图的顶点，「分子」于「分母」的比值可以表示成一个有向关系（因为「分子」和「分母」是有序的，不可以对换），并且这个图是一个带权图，values 就是对应的有向边的权值。例 1 中给出的 equations 和 values 表示的「图形表示」、「数学表示」和「代码表示」如下表所示。其中 parent[a] = b 表示：结点 a 的（直接）父亲结点是 b，与之对应的有向边的权重，记为 weight[a] = 2.0，即 weight[a] 表示结点 a 到它的 直接父亲结点 的有向边的权重。

![img](https://pic.leetcode-cn.com/1609860627-dZoDYx-image.png)

#### 「统一变量」与「路径压缩」的关系

刚刚在分析例 1 的过程中，提到了：可以把一个一个 query 中的不同变量转换成 同一个变量，这样在计算 query 的时候就可以以 O(1)O(1) 的时间复杂度计算出结果，在「并查集」的一个优化技巧中，「路径压缩」就恰好符合了这样的应用场景。

为了避免并查集所表示的树形结构高度过高，影响查询性能。「路径压缩」就是针对树的高度的优化。「路径压缩」的效果是：在查询一个结点 a 的根结点同时，把结点 a 到根结点的沿途所有结点的父亲结点都指向根结点。如下图所示，除了根结点以外，所有的结点的父亲结点都指向了根结点。特别地，也可以认为根结点的父亲结点就是根结点自己。如下国所示：路径压缩前后，并查集所表示的两棵树形结构等价，路径压缩以后的树的高度为 2，查询性能最好。

![image.png](https://pic.leetcode-cn.com/1609861184-fXdaCo-image.png)

#### 如何在「查询」操作的「路径压缩」优化中维护权值变化

如下图所示，我们在结点 a 执行一次「查询」操作。路径压缩会先一层一层向上先找到根结点 d，然后依次把 c、b 、a 的父亲结点指向根结点 d。

c 的父亲结点已经是根结点了，它的权值不用更改；
b 的父亲结点要修改成根结点，它的权值就是从当前结点到根结点经过的所有有向边的权值的乘积，因此是 3.03.0 乘以 4.0 也就是 12.0；
a 的父亲结点要修改成根结点，它的权值就是依然是从当前结点到根结点经过的所有有向边的权值的乘积，但是我们 没有必要把这三条有向边的权值乘起来，这是因为 b 到 c，c 到 d 这两条有向边的权值的乘积，我们在把 b 指向 d 的时候已经计算出来了。因此，a 到根结点的权值就等于 b 到根结点 d 的新的权值乘以 a 到 b 的原来的有向边的权值。

![image.png](https://pic.leetcode-cn.com/1609861645-DbxMDs-image.png)

#### 如何在「合并」操作中维护权值的变化

「合并」操作基于这样一个 很重要的前提：我们将要合并的两棵树的高度最多为 22，换句话说两棵树都必需是「路径压缩」以后的效果，两棵树的叶子结点到根结点最多只需要经过一条有向边。

![image.png](https://pic.leetcode-cn.com/1609862151-XZgKGY-image.png)

![image.png](https://pic.leetcode-cn.com/1609862263-LAsiiW-image.png)

并查集的「查询」操作会执行「路径压缩」，所以真正在计算两个变量的权值的时候，绿色结点已经指向了根结点，和黄色结点的根结点相同。因此可以用它们指向根结点的有向边的权值的比值作为两个变量的比值。



![image.png](https://pic.leetcode-cn.com/1609862467-jtZvlE-image.png)

```c++
class Solution {
public:
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
            vector<double>res;
            int n = equations.size();
            //字符串和id一一对应
            unordered_map<string,int>hash;
            int id = 0;
            for(int i = 0; i<n; i++){
                if(hash.find(equations[i][0]) == hash.end()){
                    hash[equations[i][0]] = id++;
                }
                if(hash.find(equations[i][1]) == hash.end()){
                    hash[equations[i][1]] = id++;
                }
            }
            //每个节点的根节点 初始化时根节点都是自己
            vector<int>roots(id);
            for(int i = 0; i<id; i++){
                roots[i] = i;
            }

            //节点指向根节点的权值 a->b a/b = 3.0
            vector<double>weights(id,1.0);
            //并查集合并
            for(int i = 0; i<n; i++){
                merge(roots,weights,hash[equations[i][0]],hash[equations[i][1]],values[i]);
            }
            //查询操作
            int  m = queries.size();
            for(int i = 0; i<m; i++){
                //如果查询的字符串在equations 中没有出现过
                if(hash.find(queries[i][0]) == hash.end() || hash.find(queries[i][1]) == hash.end())res.push_back(-1.0);
                else{
                    int id1 = hash[queries[i][0]];
                    int id2 = hash[queries[i][1]];
                    int root1 = findroot(roots,weights,id1);
                    int root2 = findroot(roots,weights,id2);

                    //判断两个节点是不是属于一个并查集
                    if(root1 == root2){
                        res.push_back(weights[id1]/weights[id2]);

                    }else{
                        res.push_back(-1.0);
                    }
                }
            }

            return res;
    }
    void merge(vector<int>&roots,vector<double>&weights, int id1,int id2, double value){
        int root1 = findroot(roots,weights,id1);
        int root2 = findroot(roots,weights,id2);
        //如果根节点相同 属于同一个并查集
        if(root1 == root2){
            return;
        }
        roots[root1] = root2;
        //更新对应的权值id
        weights[root1] = value*weights[id2]/weights[id1];

    }

    //查找并查集根节点 伴随路径压缩

    int findroot(vector<int>&roots, vector<double>&weights,int id){
        //路径压缩
        if(roots[id]!=id){
            // a/c = a/b/c;
            int parent = findroot(roots,weights,roots[id]);
            weights[id] *= weights[roots[id]];
            roots[id] = parent;

        }
        return roots[id];
    }
};
```

# 单调栈

#### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg)

```
输入： heights = [2,4]
输出： 4
```

先对柱子插入最左侧插入0，就不需要讨论了

```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        //以空间换时间 利用单调栈来记录宽度
        stack<int>st;
        int ans = 0;
        //初始位置插入0
        heights.insert(heights.begin(),0);
        heights.push_back(0);
        for(int i = 0; i<heights.size(); i++){
            while(!st.empty() && heights[st.top()]>heights[i]){
                int cur = st.top();
                st.pop();
                
                if(!st.empty()){
                    int left = st.top()+1;
                    int right = i-1;
                    ans = max(ans,(right - left+1)*heights[cur]);

                }
            
            
            }
            st.push(i);
        }
        return ans;
        
    }
};
```

不在最开始添加元素

```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        //以空间换时间 利用单调栈来记录宽度
        stack<int>st;
        int ans = 0;
        // heights.insert(heights.begin(),0);
        heights.push_back(0);
        for(int i = 0; i<heights.size(); i++){
            while(!st.empty() && heights[st.top()]>heights[i]){
                //遍历每一个高度 求对应最大面积
               int h = heights[st.top()];
               st.pop();
               if(st.empty())
                    ans = max(ans,i*h);
               else ans = max(ans,(i-st.top()-1)*h);
            
            
            }
            st.push(i);
        }
        return ans;
        
    }
};
```

#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)



# 数学

#### [5253. 找到指定长度的回文数](https://leetcode-cn.com/problems/find-palindrome-with-fixed-length/)

给你一个整数数组 queries 和一个 正 整数 intLength ，请你返回一个数组 answer ，其中 answer[i] 是长度为 intLength 的 正回文数 中第 queries[i] 小的数字，如果不存在这样的回文数，则为 -1 。

回文数 指的是从前往后和从后往前读一模一样的数字。回文数不能有前导 0 。

```
输入：queries = [1,2,3,4,5,90], intLength = 3
输出：[101,111,121,131,141,999]
解释：
长度为 3 的最小回文数依次是：
101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, ...
第 90 个长度为 3 的回文数是 999 。

输入：queries = [2,4,6], intLength = 4
输出：[1111,1331,1551]
解释：
长度为 4 的前 6 个回文数是：
1001, 1111, 1221, 1331, 1441 和 1551 。


```

本题属于数学找规律了。

第一次直接采用暴力循环，找出每一个回文数。这种方法直接超时了。

本题按照queries数组作为下标找回文，那么先总结到回文串的公式。

![image-20220327165526277](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20220327165526277.png)

只需要找前半部分，后半部分为前部分的反转.

base是对应的intlenth的最长回文个数。超出了这个个数肯定就找不到，返回-1即可。利用字符串更好进行操作。

```c++
class Solution {
public:
    vector<long long> kthPalindrome(vector<int>& queries, int intLength) {
        vector<long long>res;
        int base = pow(10,(intLength-1)/2);
        int size = 9*pow(10,(intLength-1)/2);
        for(auto &q:queries){
            int number = base+q-1;
            if(q>size){
                res.emplace_back(-1);
                continue;
            }
            string strLeft = to_string(number);
            string strRight = strLeft;
            reverse(strRight.begin(),strRight.end());
           
            if(intLength%2 ==0){
                strLeft+=strRight;
                res.emplace_back(stol(strLeft));
            }else{
                strLeft.erase(strLeft.end()-1);
                strLeft+=strRight;
                res.emplace_back(stol(strLeft));
            }


        }
        return res;
        
    }
};
```

