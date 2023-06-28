import functools
import math
import heapq
from msilib.schema import IniFile
from ast import Str
from typing import Optional, List
from collections import defaultdict, deque
from prometheus_client import Counter 
from regex import W


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    # Function to add a new node
    def pushNode(self, head_ref, data_val):

        # Allocate node and put in the data
        new_node = ListNode(data_val)

        # Link the old list off the new node
        new_node.next = head_ref

        # move the head to point to the new node
        head_ref = new_node
        return head_ref

    # A utility function to print a given linked list
    def printNode(self, head):
        while head != None:
            print("%d->" % head.val, end="")
            head = head.next
        print("NULL")

    def getLen(self, head):
        temp = head
        len = 0

        while temp != None:
            len += 1
            temp = temp.next
        return len


class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.end = False


class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        curr = self.root
        for c in word:
            i = ord(c) - ord("a")
            if not curr.children[i]:
                curr.children[i] = TrieNode()
            curr = curr.children[i]
        curr.end = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        curr = self.root
        for c in word:
            i = ord(c) - ord("a")
            if not curr.children[i]:
                return False
            curr = curr.children[i]
        return curr.end

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        curr = self.root
        for c in prefix:
            i = ord(c) - ord("a")
            if not curr.children[i]:
                return False
            curr = curr.children[i]
        return True


class MinStack:
    def __init__(self):
        self.stack = []
        self.minstack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(
            val, self.minstack[-1] if self.minstack else val
        )  # if stack is not empty and if yes pick the last one
        self.minstack.append(val)

    def pop(self) -> None:
        if len(self.stack) >= 0:
            self.stack.pop()
            self.minstack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minstack[-1]


class SLinkedList:
    def __init__(self):
        self.headval = None

    # Print the linked list
    def listprint(self):
        printval = self.headval
        while printval is not None:
            print(printval.dataval)
            printval = printval.nextval

    def AtBegining(self, newdata):
        NewNode = Node(newdata)
        # Update the new nodes next val to existing node
        NewNode.nextval = self.headval
        self.headval = NewNode


# list = SLinkedList()
# list.headval = Node("Mon")
# e2 = Node("Tue")
# e3 = Node("Wed")

# list.headval.nextval = e2
# e2.nextval = e3

# list.AtBegining("Sun")
# list.listprint()


class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None


class Node_n:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        # minHeap w/ K largest integers
        self.minHeap, self.k = nums, k
        heapq.heapify(self.minHeap)
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)

    def add(self, val: int) -> int:
        heapq.heappush(self.minHeap, val)
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        return self.minHeap[0]


class Solutions:
    def longestCommonPrefix(self, strs):

        if not strs:
            return ""
        shortest = min(strs, key=len)
        for i, ch in enumerate(shortest):
            for other in strs:
                if other[i] != ch:
                    return shortest[:i]
        return shortest

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        digits = "123456789"
        res = []
        nl = len(str(low))
        nh = len(str(high))

        for i in range(nl, nh + 1):
            for j in range(0, 10 - i):
                num = int(digits[j:j + i])
                if num >= low and num <= high:
                    res.append(num)
        return res

    def combine(self, n: int, k: int) -> List[List[int]]:
        # will try to use dfs at first as it's about the recursion
        def dfs(first=1, nlist=[]):
            if len(nlist) == k:
                return listn.append(nlist.copy())
            for i in range(first, n + 1):
                nlist.append(i)
                dfs(i + 1, nlist)
                nlist.pop()

        listn = []
        dfs()
        return list

    def pivotIndex(self, nums: List[int]) -> int:
        leftSum, rightSum = 0, sum(nums)
        if sum(nums[1:]) == 0:
            return 0
        for i, n in enumerate(nums):
            rightSum -= n
            if leftSum == rightSum:
                return i
            leftSum += n
        return -1

    def runningSum(self, nums: List[int]) -> List[int]:
        arrnum = [nums[0]]
        for i, j in zip(nums[1:], arrnum):
            arrnum.append(i + j)
        return arrnum

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if target < 1:
            return "Wrong param!"

        # We will try to make dfs
        def dfs(item, current, proceded_sum):
            if proceded_sum == target:
                if current[:] not in res:
                    res.append(current[:])
                return
            if item >= len(candidates) or proceded_sum > target:
                return
            current.append(candidates[item])
            dfs(item, current, proceded_sum + candidates[item])
            current.pop()
            dfs(item + 1, current, proceded_sum)

        res = []
        dfs(0, [], 0)
        return res

    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False

        slist = list()
        for idx in s:
            slist.append(s.index(idx))
        tlist = list()
        for idx in t:
            tlist.append(t.index(idx))
        if slist == tlist:
            return True
        return False

        # hasheds = {}
        # for ind, ch in enumerate(s):
        #     if ch in hasheds:
        #         hasheds[ch].add(ind)
        #     else:
        #         hasheds[ch] = {
        #             ind,
        #         }
        # hashedt = {}
        # for ind, ch in enumerate(t):
        #     if ch in hashedt:
        #         hashedt[ch].add(ind)
        #     else:
        #         hashedt[ch] = {
        #             ind,
        #         }
        # return tuple(hasheds.values()) == tuple(hashedt.values())

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        for ch in s:
            chSearch = t.find(ch)
            if chSearch == -1:
                return False
            else:
                t = t[chSearch + 1:]
        return True

        # t = "".join([tt for tt in t if tt in s])
        # if t.find(s) != -1:
        #     return True
        # return False

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:

        # Need to sort first for the  help to find duplicates while traversing the tree
        candidates.sort()

        def dfs(item, current, target):
            if target == 0:  # found target
                res.append(current[:])
            if target <= 0:
                return

            # to except diplicates
            previous = -1  # to except diplicates
            for i in range(item, len(candidates)):
                if candidates[i] == previous:
                    continue

                current.append(candidates[i])
                dfs(i + 1, current, target - candidates[i])
                current.pop()
                previous = candidates[i]

        res = []
        dfs(0, [], target)
        return res

    def sequentialDigits2(self, low: int, high: int) -> List[int]:
        possible_digits = "123456789"
        res = []
        nl, nh = len(str(low)), len(str(high))

        for i in range(nl, nh + 1):
            for j in range(0, 10 - i):
                num = int(possible_digits[j: j + 1])
                if num < low and num <= high:
                    res.append(num)
        print(res)
        return res

    def permute(self, list, s):
        if list == 1:
            return s
        else:
            return [
                y + x for y in self.permute(1, s) for x in self.permute(list - 1, s)
            ]

    def bubblesort(list):
        # Swap the elements to arrange in order
        for iter_num in range(len(list) - 1, 0, -1):
            for idx in range(iter_num):
                if list[idx] > list[idx + 1]:
                    temp = list[idx]
                    list[idx] = list[idx + 1]
                    list[idx + 1] = temp

    def merge_sort(self, unsorted_list):
        if len(unsorted_list) <= 1:
            return unsorted_list
        # Find the middle point and devide it
        middle = len(unsorted_list) // 2
        left_list = unsorted_list[:middle]
        right_list = unsorted_list[middle:]

        left_list = self.merge_sort(left_list)
        right_list = self.merge_sort(right_list)
        return list(self.merge(left_list, right_list))

        # Merge the sorted halves
        def merge(left_half, right_half):
            res = []
            while len(left_half) != 0 and len(right_half) != 0:
                if left_half[0] < right_half[0]:
                    res.append(left_half[0])
                    left_half.remove(left_half[0])
                else:
                    res.append(right_half[0])
                    right_half.remove(right_half[0])
            if len(left_half) == 0:
                res = res + right_half
            else:
                res = res + left_half
            return res

    def insertion_sort(InputList):
        for i in range(1, len(InputList)):
            j = i - 1
            nxt_element = InputList[i]

            # Compare the current element with next one
            while (InputList[j] > nxt_element) and (j >= 0):
                InputList[j + 1] = InputList[j]
                j = j - 1
            InputList[j + 1] = nxt_element

    def selection_sort(input_list):
        for idx in range(len(input_list)):
            min_idx = idx
            for j in range(idx + 1, len(input_list)):
                if input_list[min_idx] > input_list[j]:
                    min_idx = j
        # Swap the minimum value with the compared value
        input_list[idx], input_list[min_idx] = input_list[min_idx], input_list[idx]
        return input_list

    test_list = [19, 2, 31, 45, 30, 11, 121, 27]
    sorted_l = selection_sort(test_list)
    print(sorted_l)

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        hashset = {}

        for index, val in enumerate(nums):
            if val in hashset:
                if abs(index - hashset[val]) <= k:
                    return True
            hashset[val] = index
        return False

    def isAnagram(self, s: str, t: str) -> bool:

        countS, countT = {}, {}
        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        for c in countS:
            if countS[c] != countT.get(c, 0):
                return False
        return True

    def removeAnagrams(self, words: List[str]) -> List[str]:
        index = 0
        while index < len(words) - 1:
            if sorted(words[index]) == sorted(words[index + 1]):
                words.pop(index + 1)
            else:
                index += 1
        return words

    def findAnagrams(self, s: str, p: str) -> List[int]:
        hashmap = defaultdict(int)
        res = []
        plen = len(p)
        slen = len(s)

        if plen > slen:
            return []

        # map p letters
        for ch in p:
            hashmap[ch] += 1

        # go through the window
        for i in range(plen - 1):
            if s[i] in hashmap:
                hashmap[s[i]] -= 1

        # slide the window from the end
        for i in range(-1, slen - plen + 1):
            if i > -1 and s[i] in hashmap:
                hashmap[s[i]] += 1
            if i + plen < slen and s[i + plen] in hashmap:
                hashmap[s[i + plen]] -= 1

            # check whether we found an anagram
            if all(v == 0 for v in hashmap.values()):
                res.append(i + 1)

        return res
        # Not optimised
        # res = []
        # first = 0
        # second = first + len(p)
        # while first < len(s):
        #     if s[first] in p:
        #         if sorted(s[first:second]) == sorted(p):
        #             res.append(first)
        #     first += 1
        #     second += 1
        # return res

    def twoSum(self, nums: List[int], target: int) -> List[int]:

        for index, value in enumerate(nums):
            print(nums[index + 1:])
            if target - value in nums[index + 1:]:
                print(len(nums))
                return (
                    [0, 1]
                    if len(nums) == 2
                    else [index, nums[index + 1:].index(target - value)]
                )
        return

    def twoSum2(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1

        while left < right:
            cur_sum = numbers[left] + numbers[right]

            if cur_sum > target:
                right -= 1
            elif cur_sum < target:
                left += 1
            else:
                return [left + 1, right + 1]

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashmap = {}
        res = []

        for s in strs:
            if str(sorted(s)) in hashmap:
                hashmap[str(sorted(s))].append(s)
            else:
                hashmap[str(sorted(s))] = [s]
        for anagram in hashmap:
            res.append(hashmap[anagram])
        return res

    def isHappy(self, n: int) -> bool:
        if n < 0:
            return False

        hset = set()  # hset for check looping
        while n != 1:
            if n in hset:
                return False
            else:
                hset.add(n)
                n = sum([int(d) ** 2 for d in str(n)])
        return True

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hashmap = {}
        freq = [[] for i in range(len(nums) + 1)]  # bucket for our nums
        res = []

        for n in nums:  # count numbers
            hashmap[n] = 1 + hashmap.get(n, 0)
        for num, count in hashmap.items():
            freq[count].append(num)

        for i in range(len(freq) - 1, 0, -1):

            for n in freq[i]:
                res.append(n)
            if len(res) == k:
                return res

    def topKFrequent2(self, words: List[str], k: int) -> List[str]:
        wordsC = Counter(sorted(words))
        top_k = wordsC.most_common(k)
        return [s[0] for s in top_k]

    def isPalindrome(self, s: str) -> bool:
        start = 0
        end = len(s) - 1

        while start < end:
            while start < end and not self.is_alphanumeric_low(s[start]):
                start += 1
            while end > start and not self.is_alphanumeric_low(s[end]):
                end -= 1
            if s[start].lower() != s[end].lower():
                print(f"{s[start]}!= {s[end]}")
                return False

            start, end = start + 1, end - 1
        return True

    def is_alphanumeric_low(self, symb):
        return (
            (ord("A") <= ord(symb) <= ord("Z"))
            or (ord("a") <= ord(symb) <= ord("z"))
            or (ord("0") <= ord(symb) <= ord("9"))
            and symb.islower()
        )

    def validPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1

        while left <= right:
            if s[left].lower() != s[right].lower():
                return self.check_palindrome(
                    s, left, right - 1
                ) or self.check_palindrome(s, left + 1, right)
            left, right = left + 1, right - 1

        return True

    def check_palindrome(s, left, right):
        while left < right:
            if s[left].lower() != s[right].lower():
                return False
            left, right = left + 1, right - 1

        return True

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        for i, a in enumerate(nums):
            if i > 0 and a == nums[i - 1]:
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                three_sum = a + nums[left] + nums[right]
                if three_sum > 0:
                    right -= 1
                elif three_sum < 0:
                    left += 1
                else:
                    res.append([a, nums[left], nums[right]])
                    while nums[left] == nums[left - 1] and left < right:
                        left += 1
        return True

    def maxProfit(self, prices: List[int]) -> int:
        # sliding window
        # left-buy, right-sell
        left, right = 0, 1
        max_profit = 0

        while right < len(prices):
            # if it is profitable
            if prices[left] < prices[right]:
                profit = prices[right] - prices[left]
                # if it is maximum profitable
                max_profit = max(max_profit, profit)
            else:
                left = right
            right += 1
        return max_profit

    def lengthOfLongestSubstring(self, s: str) -> int:
        set_substr = set()
        left = 0
        res = 0

        for r in range(len(s)):
            while s[r] in set_substr:
                set_substr.remove(s[left])
                left += 1
            set_substr.add(s[r])
            res = max(res, r - left + 1)  # compare  withcurrent sliding window
        return res

    def characterReplacement(self, s: str, k: int) -> int:
        # using sliding window and hashmap
        hashmap = {}
        left = 0
        res = 0

        for r in range(len(s)):
            hashmap[s[r]] = 1 + hashmap.get(s[r], 0)

            if (r - left + 1) - max(
                hashmap.values()
            ) > k:  # if the number of replacements we have to do is alowed
                hashmap[s[left]] -= 1
                left += 1
            res = max(res, r - left + 1)  # size of the window
        return res

    def getHint(self, secret: str, guess: str) -> str:
        from collections import Counter

        badN = Counter(secret) - Counter(guess)
        correct = 0
        for i in range(len(secret)):
            if secret[i] == guess[i]:
                correct += 1
        count = len(secret) - sum(badN.values()) - correct
        return f"{correct}A{count}B"

    def generateParenthesis(self, n: int) -> List[str]:
        # can't start with closed paranthesis and add if closed < open

        stack = []
        res = []

        def backtrack(openp, closedp):
            if openp == closedp == n:
                res.append("".join(stack))
                return
            if openp < n:
                stack.append("(")
                backtrack(openp + 1, closedp)
                stack.pop()

            if closedp < openp:
                stack.append(")")
                backtrack(openp, closedp + 1)
                stack.pop()

        backtrack(0, 0)
        return res

    def binary_search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:  # = for left pointer to pass through the right pointer
            mid = left + ((right - left) // 2)
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid
        return -1

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows, columns = len(matrix), len(matrix[0])

        # searching for row through the matrix
        top_row, bot_row = 0, rows - 1
        # row = top_row + ((bot_row - top_row) // 2) #- mid_val

        while top_row <= bot_row:

            row = (bot_row + top_row) // 2
            if target > matrix[row][-1]:  # greater than the largest value in this row
                top_row = row + 1  # look into rows with larger vlues
            elif target < matrix[row][0]:  # smaler than the smalest value in this row
                bot_row = row - 1
            else:
                break

        if not (top_row <= bot_row):  # if none of the row contain the targwt value
            return False

        # searching for the number through the founded row
        left, right = 0, columns - 1

        while left <= right:

            # mid = left + ((right - left) // 2)
            mid = (left + right) // 2
            if target > matrix[row][mid]:  # greater than the number at position middle
                left = mid + 1
            elif (
                target < matrix[row][mid]
            ):  # smaller than the smalest value in this row
                right = mid - 1
            else:
                return True
        return False

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head  # O(1)

        while cur:  # O(n)
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp

        return prev

    def middleNode(self, head_list: Optional[ListNode]) -> Optional[ListNode]:
        print(len(head_list))

        # 1
        # work_code for ListNode
        head = None
        temp = ListNode()
        for i in head_list:
            head = temp.pushNode(head, i)
            temp.printNode(head)

        if head:
            midlen = temp.getLen(head) // 2
            while midlen != 0:
                head = head.next
                midlen -= 1
            return head

        # 2
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        slow = fast = head

        for i in range(n):
            fast = fast.next
        if not fast:
            head = head.next
        else:
            while fast.next:
                fast = fast.next
                slow = slow.next
            slow.next = slow.next.next
        return head

    def reverseList_recursion(self, head: ListNode) -> ListNode:
        if not head:
            return None
        else:
            for value in reversed(head):
                head = ListNode(value, head)

        new_head = head
        if head.next:
            new_head = self.reverseList(head.next)
            head.next.next = head
        head.next = None
        return new_head

    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode()
        tail = dummy

        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next  # update pointer
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next

        if list1:
            tail.next = list1  # take remain portion of list1 and insert it to the end of the list
        elif list2:
            tail.next = list2

        return dummy.next

    def mergeTwoLists2(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if not list1:
            return list2  # for sure add remaining nodes
        if not list2:
            return list1

        if list1.val <= list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list2.next, list1)
            return list2

    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """

        slow, fast = head, head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next  # to shift by two

        # split into two different lists
        second = slow.next
        prev = slow.next = None

        # reverse the second portion on the list
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp

        # merge two lists
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1  # insert between  two nodes in non-reversed list
            first = tmp1
            second = tmp2

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:  # find the cycle
                while slow != head:
                    slow, head = slow.next, head.next  # find the meet of slow and head
                return slow
        return None

    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0

        # define where this nodes intersect
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        slow2 = 0
        while True:
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow == slow2:
                return slow

    # Tree traversal
    # N-ary Tree Preorder Traversal
    def preorder(self, root: "Node") -> List[int]:
        res = []

        def dfs(root, res):
            if root:
                res.append(root.val)
                for child in root.children:
                    dfs(child, res)

        dfs(root, res)
        return res

    # DFS
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        # swap the children
        tmp = root.left
        root.left = root.right
        root.right = tmp

        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    def maxDepth(self, root: Optional[TreeNode]) -> int:

        if not root:
            return 0

        # recursive DFS
        # return 1+max(self.maxDepth(root.left),self.maDepth(root.right))
        # iterative pre-order DFS (Stack)
        stack = [[root, 1]]
        res = 0

        while stack:
            node, depth = stack.pop()

            if node:
                res = max(res, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])

        # return res

        # iterative BFS (Queue)
        level = 0
        q = deque([root])
        while q:

            for _ in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            level += 1

        return level

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # find from bottom up
        res = [0]

        def dfs(root):
            if not root:
                return -1

            left = dfs(root.left)
            right = dfs(root.right)
            res[0] = max(res[0], 2 + left + right)  # diameter. 2 edges
            return 1 + max(left, right)  # high

        dfs(root)
        return res[0]

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # recursevly botom up

        def dfs(root):
            if not root:
                return [True, 0]

            left, right = dfs(root.left), dfs(root.right)

            balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1

            return [balanced, 1 + max(left[1], right[1])]

        return dfs(root)[0]

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # if not p and not q: return True #  if both are empty
        # if not p or not q or p.val != q.val: return False
        # return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

        if p and q and p.val == q.val:  # if their subrties are equal
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not subRoot:
            return True
        if not root:
            return False

        # is Subtree
        if self.sameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def sameTree(self, s, t):
        if not s and not t:
            return True
        if s and t and s.val == t.val:
            return self.sameTree(s.left, t.left) and self.sameTree(s.right, t.right)
        return False

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid(node, left, right):
            if not node:
                return True
            if not (left < node.val < right):
                return False

            return valid(node.left, left, node.val) and valid(
                node.right, node.val, right
            )

        return valid(root, float("-inf"), float("inf"))

    def isValidBST2(self, root: Optional[TreeNode]) -> bool:
        def validate(node, left, right):
            """each iteration we move each of elements
            from left to right, than from right to left"""
            if not node:
                return True
            if left is not None and node.val <= left:
                return False
            if right is not None and node.val >= right:
                return False
            return validate(node.left, left, node.val) and validate(
                node.right, node.val, right
            )

        return validate(root, None, None)

    def findKthLargest(self, nums: List[int], k: int) -> int:
        # nums.sort()
        # return nums[len(nums) - k]

        k = len(nums) - k

        def quickSelect(left, right):
            if left == right:
                return nums[left]

            pivot, p = nums[right], left
            for i in range(left, right):
                if nums[i] <= pivot:
                    nums[p], nums[i] = nums[i], nums[p]
                    p += 1
            nums[p], nums[right] = nums[right], nums[p]

            if p > k:
                return quickSelect(left, p - 1)
            elif p < k:
                return quickSelect(p + 1, right)
            else:
                return nums[p]

        return quickSelect(0, len(nums) - 1)

    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [
            -s for s in stones
        ]  # simulation of max heap because in python  we have only min heap
        heapq.heapify(stones)

        while len(stones) > 1:
            first = heapq.heappop(stones)
            second = heapq.heappop(stones)
            if second > first:
                heapq.heappush(stones, first - second)

        stones.append(0)  # if our heap is empty
        return abs(stones[0])

    def combinationSum3(self, candidates: List[int], target: int) -> List[List[int]]:
        if target < 1:
            return []
        
        # We will try to make dfs
        def dfs(item, current, proceded_sum):
            if proceded_sum == target:
                if current[:] not in res:
                    res.append(current[:])
                    return
            if item >= len(candidates) or proceded_sum > target:
                return
            current.append(candidates[item])
            dfs(item, current, proceded_sum + candidates[item])
            current.pop()
            dfs(item + 1, current, proceded_sum)

        res = []
        dfs(0, [], 0)
        return res

    # def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:

    #     # Need to sort first for the  help to find duplicates while traversing the tree
    #     candidates.sort()

    #     def dfs(item, current, target):
    #         if target == 0:  # found target
    #             res.append(current[:])
    #         if target <= 0:
    #             return

    #         # to except duplicates
    #         previous = -1  # to except diplicates
    #         for i in range(item, len(candidates)):
    #             if candidates[i] == previous:
    #                 continue

    #             current.append(candidates[i])
    #             dfs(i + 1, current, target - candidates[i])
    #             current.pop()
    #             previous = candidates[i]

    #     res = []
    #     dfs(0, [], target)
    #     return res

    def partition(self, s: str) -> List[List[str]]:
        res, part = [], []

        def dfs(i):
            if i >= len(s):
                res.append(part.copy())
                return
            for j in range(i, len(s)):
                if self.isPali(s, i, j):
                    part.append(s[i:j + 1])
                    dfs(j + 1)
                    part.pop()

        dfs(0)
        return res

    def isPali(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left, right = left + 1, right - 1
        return True

    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        digitToChar = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "qprs",
            "8": "tuv",
            "9": "wxyz",
        }

        def backtrack(i, curStr):
            if len(curStr) == len(digits):
                res.append(curStr)
                return
            for c in digitToChar[digits[i]]:
                backtrack(i + 1, curStr + c)

        if digits:
            backtrack(0, "")

        return res

    def floodFill(
        self, image: List[List[int]], sr: int, sc: int, color: int
    ) -> List[List[int]]:

        if image[sr][sc] == color:
            return image

        def fill(r, c, col):
            if image[r][c] == col:
                image[r][c] = color
                if r > 0:  # check left raw direction
                    fill(r - 1, c, col)
                if r + 1 < len(image):  # check right raw direction
                    fill(r + 1, c, col)
                if c > 0:  # check upper column direction
                    fill(r, c - 1, col)
                if c + 1 < len(image[0]):  # check bottom column direction
                    fill(r, c + 1, col)

        fill(sr, sc, image[sr][sc])
        return image

    def orangesRotting(self, grid: List[List[int]]) -> int:
        # implement multi sorce BFS
        q = deque()
        fresh = 0
        time = 0

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:  # are there a fresh oranges?
                    fresh += 1
                if grid[r][c] == 2:  # are there rotten oranges?
                    q.append((r, c))  # coordinates of that rotten orange

        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while fresh > 0 and q:
            for i in range(len(q)):
                r, c = q.popleft()  # pop the coordinates of the rotten orange

                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    # if in bounds and fresh, make rotten
                    # and add to q
                    if (
                        row in range(len(grid))
                        and col in range(len(grid[0]))
                        and grid[row][col] == 1
                    ):

                        grid[row][col] = 2
                        q.append((row, col))
                        fresh -= 1
            time += 1
        return time if fresh == 0 else -1

    def cloneGraph(self, node: "Node_n") -> "Node_n":
        oldToNew = {}

        def dfs(node):
            if node in oldToNew:
                return oldToNew[node]

            copy = Node_n(node.val)
            oldToNew[node] = copy
            for nei in node.neighbors:
                copy.neighbors.append(dfs(nei))
            return copy

        return dfs(node) if node else None

    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0

        islands = 0
        visit = set()
        rows, cols = len(grid), len(grid[0])

        # DFS
        def dfs(r, c):
            if (
                r not in range(rows)
                or c not in range(cols)
                or grid[r][c] == "0"
                or (r, c) in visit
            ):
                return

            visit.add((r, c))
            directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dr, dc in directions:
                dfs(r + dr, c + dc)

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visit:
                    islands += 1
                    dfs(r, c)
        return islands

        # BFS
        def bfs(r, c):
            import collections
            q = collections.deque()
            visit.add((r, c))
            q.append((r, c))

            while q:
                row, col = q.popleft()
                directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

                for dr, dc in directions:
                    if (
                        (r + dr) in range(rows)
                        and (c + dc) in range(cols)
                        and grid[r + dr][c + dc] == "1"
                        and (r + dr, c + dc) not in visit
                    ):
                        q.append((r + dr, c + dc))
                        visit.add((r + dr, c + dc))

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visit:
                    bfs(r, c)
                    islands += 1
        # Binary Tree Level Order Traversal

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return

        queue = []
        res = []

        queue.append(root)
        while len(queue) > 0:
            inner_list = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                if node:
                    inner_list.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)
            if inner_list:
                res.append(inner_list)
        return res

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        visit = set()

        def dfs(r, c):
            if (
                r < 0
                or r == ROWS
                or c < 0
                or c == COLS
                or grid[r][c] == 0
                or (r, c) in visit
            ):
                return 0
            visit.add((r, c))
            return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

        area = 0
        for r in range(ROWS):
            for c in range(COLS):
                area = max(area, dfs(r, c))
        return area

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # DFS
        preMap = {i: [] for i in range(numCourses)}

        # map each course to : prereq list
        for crs, pre in prerequisites:
            preMap[crs].append(pre)

        visiting = set()

        def dfs(crs):
            if crs in visiting:
                return False
            if preMap[crs] == []:
                return True

            visiting.add(crs)
            for pre in preMap[crs]:
                if not dfs(pre):
                    return False
            visiting.remove(crs)
            preMap[crs] = []
            return True

        for c in range(numCourses):
            if not dfs(c):
                return False
        return True

    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        N = len(points)
        adj = {i: [] for i in range(N)}  # i : list of [cost, node]
        for i in range(N):
            x1, y1 = points[i]
            for j in range(i + 1, N):
                x2, y2 = points[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                adj[i].append([dist, j])
                adj[j].append([dist, i])

        # Prim's
        res = 0
        visit = set()
        minH = [[0, 0]]  # [cost, point]
        while len(visit) < N:
            cost, i = heapq.heappop(minH)
            if i in visit:
                continue
            res += cost
            visit.add(i)
            for neiCost, nei in adj[i]:
                if nei not in visit:
                    heapq.heappush(minH, [neiCost, nei])
        return res

    def climbStairs(self, n: int) -> int:
        one, two = 1, 1

        for i in range(n - 1):
            tmp = one
            one = one + two
            two = tmp
        return one

        if n <= 3:
            return n
        n1, n2 = 2, 3

        for i in range(4, n + 1):
            temp = n1 + n2
            n1 = n2
            n2 = temp
        return n2

    def fib(self, n: int) -> int:
        dic = {0: 0, 1: 1}  # firstly, we initialize the base cases

        for x in range(2, n + 2):
            if x not in dic:
                # calculate the sum of 2 previous numbers
                # and check if the value in dictionary where we store the index and fibonacci number
                dic[x] = dic[x - 1] + dic[x - 2]
        return dic[n]

    # @cache
    def fib_cached(self, n: int) -> int:
        return self.fib(n - 2) + self.fib(n - 1) if n >= 2 else n

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost.append(0)  # for not reach through the end

        for i in range(len(cost) - 3, -1, -1):
            cost[i] += min(cost[i + 1], cost[i + 2])

        return min(cost[0], cost[1])

    def rob(self, nums: List[int]) -> int:

        rob1, rob2 = 0, 0  # scenario

        # [rob1, rob2, n, n+1, ... ]
        for n in nums:
            temp = max(n + rob1, rob2)  # we only can deal with [n+rob1] or rob2
            rob1 = rob2  # to move to next position
            rob2 = temp
        return rob2

    def rob2(self, nums: List[int]) -> int:
        return max(
            nums[0], self.helper(nums[1:]), self.helper(nums[:-1])
        )  # if only house, skip the first house, skip the last index

    def helper(self, nums):
        rob1, rob2 = 0, 0

        for n in nums:
            newRob = max(rob1 + n, rob2)
            rob1 = rob2  # previous
            rob2 = newRob  # previous
        return rob2

    def countSubstrings(self, s: str) -> int:
        res = 0

        for i in range(len(s)):
            res += self.countPali(s, i, i)
            res += self.countPali(s, i, i + 1)
        return res

    def countPali(self, s, left, right):
        res = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            res += 1
            left -= 1
            right += 1
        return res

    def longestPalindrome(self, s: str) -> str:
        """
        Starting in the middle. Expending outwords <-- -->
        """

        res = ""
        resLen = 0

        for i in range(len(s)):
            # odd length
            left, right = i, i
            while (
                left >= 0 and right < len(s) and s[left] == s[right]
            ):  # starting in the middle and expending outwords
                if (
                    right - left + 1
                ) > resLen:  # lenght of the palindrome is graeter when the current
                    res = s[left:right + 1]
                    resLen = right - left + 1
                left -= 1  # expend  to the left
                right += 1  # expend  to the right

            # even length
            left, right = i, i + 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                if (right - left + 1) > resLen:
                    res = s[left:right + 1]
                    resLen = right - left + 1
                left -= 1
                right += 1
        return res

    def longestPalindrome2(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == s.count(s[0]):
            return len(s)
        odd_s_hash = {
            syl: s.count(syl) % 2 != 0 for syl in s if (s.count(syl) % 2 != 0)
        }
        if len(odd_s_hash) > 1:
            return len(s) - (len(odd_s_hash) - 1)
        else:
            return len(s)

    def numDecodings(self, s: str) -> int:
        # Memoization
        dp = {len(s): 1}

        def dfs(i):
            if i in dp:  # is i cashed or end of the string
                return dp[i]

            if s[i] == "0":
                return 0

            res = dfs(i + 1)
            if i + 1 < len(s) and (
                s[i] == "1"
                or s[i] == "2"  # if we have chararcter next
                and s[i + 1] in "0123456"
            ):
                res += dfs(i + 2)
            dp[i] = res  # cash
            return res

        return dfs(0)

        # # Dynamic Programming O(1)
        dp = {len(s): 1}
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]

            if i + 1 < len(s) and (
                s[i] == "1" or s[i] == "2" and s[i + 1] in "0123456"
            ):
                dp[i] += dp[i + 2]
        return dp[0]

    def uniquePaths(self, m: int, n: int) -> int:
        # using memoisation
        def dfs(i, j):
            if i >= m or j >= n:
                return 0
            if i == m - 1 and j == n - 1:
                return 1
            return dfs(i + 1, j) + dfs(i, j + 1)

        return dfs(0, 0)

    def uniquePaths2(self, m, n):
        from itertools import product

        # with help of tabulation
        dp = [[1] * n for i in range(m)]
        for i, j in product(range(1, m), range(1, n)):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def maxSubArray(self, nums: List[int]) -> int:
        # sliding wondow
        max_sum = nums[0]  # At even one number  is in the nums
        cur_sum = 0

        for n in nums:
            if cur_sum < 0:  # if we meet negative we skip it
                cur_sum = 0
            cur_sum += n
            max_sum = max(cur_sum, max_sum)
        return max_sum

    def maxProduct(self, nums: List[int]) -> int:
        # O(n)/O(1) : Time/Memory
        res = max(nums)
        curMin, curMax = 1, 1

        for n in nums:
            if n == 0:
                curMin, curMax = 1, 1
                continue
            tmp = curMax * n
            curMax = max(n * curMax, n * curMin, n)
            curMin = min(tmp, n * curMin, n)
            res = max(res, curMax)
        return res

    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) - 1

        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= goal:
                goal = i  # shift the goal closer
        return goal == 0

        for n in range(len(nums)):
            print(nums[n])
            n += nums[n]
            if len(nums) - n < nums[n]:
                return False
        return True

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        start, end = len(gas) - 1, 0
        total = gas[start] - cost[start]

        while start >= end:
            while total < 0 and start >= end:
                start -= 1
                total += gas[start] - cost[start]
            if start == end:
                return start
            total += gas[end] - cost[end]
            end += 1
        return -1

    def canCompleteCircuit2(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1

        total = 0
        res = 0

        for i in range(len(gas)):
            total += gas[i] - cost[i]
            if total < 0:
                total = 0
                res = i + 1
        return res

    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        good = set()

        for t in triplets:
            if t[0] > target[0] or t[1] > target[1] or t[2] > target[2]:
                continue
            for i, v in enumerate(t):
                if v == target[i]:
                    good.add(i)
        return len(good) == 3

    def canAttendMeetings(self, intervals):
        intervals.sort(key=lambda i: i.start)

        for i in range(1, len(intervals)):
            i1 = intervals[i - 1]
            i2 = intervals[i]

            if i1.end > i2.start:
                return False
        return True

    def miniMaxSum(self, arr):
        # Write your code here
        arr = sorted(arr)
        min = str(sum(arr[0:4]))
        max = str(sum(arr[len(arr):0: -1]))
        print(f"{min} {max}")

    def backspaceCompare(self, s: str, t: str) -> bool:
        from collections import deque

        s = deque(s)
        t = deque(t)

        def removeEmpty(st) -> set:
            s_chToDel = 0
            sres = []
            while st:
                spop = st.pop()
                if spop == "#":
                    s_chToDel += 1
                else:
                    if s_chToDel < 1:
                        sres.append(spop)
                    else:
                        if st:
                            s_chToDel -= 1
            return sres

        return removeEmpty(s) == removeEmpty(t)

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # let's try sliding window

        # for i in range(len(nums)):
        #     output.append(math.prod(nums)-nums[i])
        # print(output)

        # q = deque(nums)
        # output = []
        # for i in range(len(nums)):
        #     poped = q.popleft()
        #     output.append( math.prod(list(q)))
        #     q.append(poped)
        # print(output)

        res = [1] * (len(nums))

        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res

    def BinarySearch(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:  # = for left pointer to pass through the right pointer
            mid = left + ((right - left) // 2)
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid

    def isBadVersion(self, version: int) -> bool:
        # ver = {
        #         1 : False,
        #         2 : False,
        #         3 : False,
        #         4 : True,
        #         5 : True,
        #         }
        ver = {
            1: False,
            2: True,
        }
        return ver.get(version, 0)

    def firstBadVersion(self, n: int) -> int:

        left, right = 1, n

        bad_version = 0
        while left <= right:
            mid = left + ((right - left) // 2)
            if self.isBadVersion(mid):
                bad_version = mid
                right = mid - 1
            else:
                left = mid + 1

        return bad_version

    def searchInsert(self, nums: List[int], target: int) -> int:
        # if target == 0: return 0
        # if target > nums[-1]: return len(nums)

        left, right = 0, len(nums) - 1
        while left <= right:

            mid = left + ((right - left) // 2)
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                return mid
        if target not in nums:
            if right - left == 1:
                return right + 1
            else:
                return left

    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # nums = deque(nums)
        # for i in range(k):
        #     nums.appendleft(nums.pop())
        # k2 = k + 1 if len(nums) % 2 else k

        n = len(nums)
        res = [0] * n
        for i in range(n):
            print((i + k) % n)
            res[(i + k) % n] = nums[i]
        nums[:] = res

        # k_shift = abs(k) % len(nums)
        # for i in range(net_shift):
        #     if k > 0:
        #         popped = nums.pop(len(nums) - 1)
        #         nums.insert(0, popped)
        #     if k < 0:
        #         popped = nums.pop(0)
        #         nums.append(popped)
        print(nums)

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == 0:
                nums.append(nums.pop(i))
        print(nums)

    def reverseWords(self, s: str) -> str:
        s_list = s.split()
        print(s_list.reverse)
        # ' '.join([reversed(ss) for ss in s_list])
        return " ".join([ss[::-1] for ss in s_list])

    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        s1Count, s2Count = [0] * 26, [0] * 26
        for i in range(len(s1)):  # through every character in s1
            s1Count[ord(s1[i]) - ord("a")] += 1
            s2Count[ord(s2[i]) - ord("a")] += 1

        matches = 0
        for i in range(26):
            matches += 1 if s1Count[i] == s2Count[i] else 0

        left = 0
        for right in range(len(s1), len(s2)):
            if matches == 26:
                return True

            index = ord(s2[right]) - ord("a")  # add  the charachter
            s2Count[index] += 1
            if s1Count[index] == s2Count[index]:
                matches += 1
            elif s1Count[index] + 1 == s2Count[index]:
                matches -= 1

            index = ord(s2[left]) - ord("a")  # remove from th eleft side
            s2Count[index] -= 1
            if s1Count[index] == s2Count[index]:
                matches += 1
            elif s1Count[index] - 1 == s2Count[index]:
                matches -= 1
            left += 1
        return matches == 26

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        def rotate_matrix(m):  # rotate the matrix counterclockwise by 90 degrees
            return [[m[row][col] for row in range(len(m))] for col in range(len(m[0])-1, -1, -1)]

        res = []
        res.extend(matrix.pop(0))
        while matrix:
            matrix = rotate_matrix(matrix)
            res.extend(matrix.pop(0))
        return res


sol = Solutions()
node = Node()
# print( sol.isPalindrome( s = "eedede"))
# print(node.maxProduct([3, -1, 4]))
# print(node.canJump([3,2,1,0,4,8]))
# my.miniMaxSum(arr = [1, 2, 3, 4, 5])
# print(my.productExceptSelf(nums=[1,2,3,4]))
# print(my.BinarySearch(nums = [-1,0,3,5,9,12], target = 9))
# print(sol.isValidBST([2,1,3]))
# print(sol.reverseList_recursion(head = [1,2,3,4,5]))
# print(sol.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]],3))
# print(sol.topKFrequent(nums = [1,1,1,2,2,3], k = 2))

# print("{.2f}".format(3.142)) # error:'float' object has no attribute '2f'

# test_typle_1 = (1, 5, 7, 3)
# test_typle_2 = (4, 6, 83, 6)

# result = map(lambda x, y: x + (y ^ 2), test_typle_1, test_typle_2)
# print(list(result))


# Sorting
def quickSort(arr, low, high):
    import random
    if low >= high:
        return
    pivot = random.choice(arr[low: high - 1])
    i = low
    j = high

    while i < j:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1
            quickSort(arr, low, j)
            quickSort(arr, i, high)


def binSearch(arr, item):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = low + (high - low)//2
        if arr[mid] > item:
            high = mid - 1
        if arr[mid] < item:
            low = mid + 1
        else:
            return mid
    return -1
