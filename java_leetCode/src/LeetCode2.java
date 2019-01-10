import java.math.BigInteger;
import java.util.*;
import java.util.concurrent.*;

public class LeetCode2 {
    static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    static class UndirectedGraphNode {
        int label;
        List<UndirectedGraphNode> neighbors;
        UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
    }

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    static class RandomListNode {
        int label;
        RandomListNode next, random;
        RandomListNode(int x) { this.label = x; }
    }

    static class Point {
        int x;
        int y;
        Point() { x = 0; y = 0; }
        Point(int a, int b) { x = a; y = b; }
    }

    public static void main(String[] args) {
        LeetCode2 leetCode2 = new LeetCode2();

        System.out.println(leetCode2.lengthOfLIS(new int[] {10,9,2,5,3,7,101,18}));
    }

    boolean isAllSame = true;
    public boolean isUnivalTree(TreeNode root) {
        if (root == null) return true;
        searchTree(root, root.val);
        return isAllSame;
    }

    private void searchTree(TreeNode node, int val) {
        if (node != null) {
            if (node.val != val) {
                isAllSame = false;
            }
            searchTree(node.left, val);
            searchTree(node.right, val);
        }
    }

    public int[] numsSameConsecDiff(int N, int K) {
        Queue<Integer> queue = new LinkedList<>();
        int[] res;
        for (int i = 1; i <= 9; i++) {
            queue.offer(i);
        }
        if (N == 1) {
            queue.offer(0);
            res = new int[queue.size()];
            int i = 0;
            for (int num : queue){
                res[i++] = num;
            }
            return res;
        }
        else {
            while (lengthOfInteger(queue.peek()) != N) {
                int value = queue.poll();
                int lastValue = value % 10;
                if (lastValue - K >= 0) {
                    queue.offer(value * 10 + lastValue - K);
                }
                if (K != 0) {
                    if (lastValue + K <= 9) {
                        queue.offer(value * 10 + lastValue + K);
                    }
                }
            }
            res = new int[queue.size()];
            int i = 0;
            for (int num : queue){
                res[i++] = num;
            }
            return res;
        }
    }

    private int lengthOfInteger(int num) {
        if (num < 0) num = -num;
        int res = 0;
        while (num != 0) {
            res ++;
            num = num/10;
        }
        return res;
    }

    String[] morses = new String[]{".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
    public int uniqueMorseRepresentations(String[] words) {
        Set<String> set = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        for (String word : words) {
            sb.setLength(0);
            for (char c : word.toCharArray()) {
                sb.append(morses[c - 'a']);
            }
            set.add(sb.toString());
        }
        return set.size();
    }

    public int repeatedNTimes(int[] A) {
        for (int i = 0; i < A.length - 2; i++) {
            if (A[i] == A[i+1] || A[i] == A[i + 2])
                return A[i];
        }
        return A[A.length - 1];
    }

    public String toLowerCase(String str) {
        char[] chars = str.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            if (c >= 'A' && c <= 'Z') {
                chars[i] = (char) (c - 'A' + 'a');
            }
        }
        return new String(chars);
    }

    public int numUniqueEmails(String[] emails) {
        Set<String> set = new HashSet<>();
        for (String email : emails) {
            set.add(formatEmail(email));
        }
        return set.size();
    }

    private String formatEmail(String email) {
        String[] parts = email.split("@");
        if (parts.length != 2) return "";
        String name = parts[0];
        name = name.replaceAll("\\.","");
        int local = -1;
        if ((local = name.indexOf('+')) != -1) {
            name = name.substring(0, local);
        }
        return name+'@'+parts[1];
    }

    public int numJewelsInStones(String J, String S) {
        int[] chars = new int[58];
        for (char c : J.toCharArray()) {
            chars[c - 65] = 1;
        }
        int count = 0;
        for (char c : S.toCharArray()) {
            if (chars[c - 65] == 1)
                count++;
        }
        return count;
    }

    public boolean isAdditiveNumber(String num) {
        int n = num.length();
        for (int i = 1; i <= n / 2; i++) {
            if (num.charAt(0) == '0' && i > 1) return false;
            BigInteger x1 = new BigInteger(num.substring(0, i));
            for (int j = 1; Math.max(j, i) <= n - i - j; ++j) {
                if (num.charAt(i) == 0 && j > 1) break;
                BigInteger x2 = new BigInteger(num.substring(i, i+j));
                if (isValid(x1, x2, j+i, num)) return true;
            }
        }
        return false;
    }

    private boolean isValid(BigInteger x1, BigInteger x2, int start, String num) {
        if (start == num.length()) return true;
        x2 = x2.add(x1);
        x1 = x2.subtract(x1);
        String sum = x2.toString();
        return num.startsWith(sum, start) && isValid(x1, x2, start + sum.length(), num);
    }

    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int[] tails = new int[nums.length];
        int size = 0;
        for (int x : nums) {
            int i = 0, j = size;
            while (i != j) {
                int m = (i + j) / 2;
                if (tails[m] < x) {
                    i = m + 1;
                }
                else
                    j = m;
            }
            tails[i] = x;
            if (i == size) size++;
        }
        return size;
    }

    public int lengthOfLIS1(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        dp[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            int value = 1;
            int max = value;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    value = dp[j] + 1;
                    max = Math.max(max, value);
                }
            }
            dp[i] = max;
        }
        int res = dp[0];
        for (int i : dp) {
            res = Math.max(res, i);
        }
        return res;
    }


    public boolean wordPattern(String pattern, String str) {
        String[] array = new String[26];
        String[] strs = str.split(" ");
        if (pattern.length() != strs.length) return false;
        Set<String> set = new HashSet<>();
        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            if (array[c - 'a'] == null) {
                array[c - 'a'] = strs[i];
                if (!set.add(strs[i])) {
                    return false;
                }
            }
            else {
                if (!array[c - 'a'].equals(strs[i]))
                    return false;
            }
        }
        return true;
    }

    public int findDuplicate(int[] nums) {
        if (nums.length > 1) {
            int slow = nums[0];
            int fast = nums[nums[0]];
            while (slow != fast) {
                slow = nums[slow];
                fast = nums[nums[fast]];
            }
            fast = 0;
            while (fast != slow) {
                fast = nums[fast];
                slow = nums[slow];
            }
            return slow;
        }
        return -1;
    }

    public void moveZeroes(int[] nums) {
        int slow = 0, fast = 0;
        while(fast <= nums.length - 1) {
            if (nums[fast] == 0) {
                fast++;
            }
            else {
                if (slow != fast)
                    nums[slow] = nums[fast];
                fast++;
                slow++;
            }
        }
        while (slow++ <= nums.length - 1) {
            nums[slow] = 0;
        }
    }

    public int numSquares(int n) {
        int[] dp = new int[n+1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            int min = Integer.MAX_VALUE;
            int j = 1;
            while (j * j <= i) {
                min = Math.min(min, dp[i - j * j] + 1);
                j++;
            }
            dp[i] = min;
        }
        return dp[n];
    }

    public int hIndex(int[] citations) {
        int length = citations.length;
        if (length == 0) return 0;
        int[] array = new int[length + 1];
        for (int i = 0; i < length; i++) {
            if (citations[i] > length)
                array[length] += 1;
            else array[citations[i]] += 1;
        }
        int t = 0;
        for (int i = length; i >= 0; i--) {
            t += array[i];
            if (t >= i) return i;
        }
        return 0;
    }

    int billion = 1000000000;
    int million = 1000000;
    int thousand = 1000;
    String Numbers[] = {"One ", "Two ", "Three ", "Four ", "Five ", "Six ", "Seven ", "Eight ", "Nine "};
    String TensNumbers[] = {"Ten ", "Twenty ", "Thirty ", "Forty ", "Fifty ", "Sixty ", "Seventy ", "Eighty ", "Ninety "};
    String TenNumber[] = {"Ten ", "Eleven ", "Twelve ", "Thirteen ", "Fourteen ", "Fifteen ", "Sixteen ", "Seventeen ", "Eighteen ", "Nineteen "};


    public String numberToWords(int num) {
        if (num == 0) return "Zero";
        StringBuilder sb = new StringBuilder();
        if (num / billion != 0) {
            sb.append(subNumberToWords(num / billion)).append("Billion ");
            num = num % billion;
        }
        if (num / million != 0) {
            sb.append(subNumberToWords(num / million)).append("Million ");
            num = num % million;
        }
        if (num / thousand != 0) {
            sb.append(subNumberToWords(num / thousand)).append("Thousand ");
            num = num % thousand;
        }
        sb.append(subNumberToWords(num));
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    private String subNumberToWords(int num) {
        if (num == 0) return "";
        StringBuilder sb = new StringBuilder();
        if (num / 100 != 0) {
            sb.append(Numbers[num / 100 - 1]).append("Hundred ");
            num = num % 100;
        }

        if (num / 10 != 0) {
            if (num / 10 == 1) {
                sb.append(TenNumber[num % 10]);
            }
            else {
                sb.append(TensNumbers[num / 10 - 1]);
                if (num % 10 != 0)
                    sb.append(Numbers[num % 10 - 1]);
            }
        }
        else {
            if(num != 0) sb.append(Numbers[num - 1]);
        }
        return sb.toString();
    }

    public int missingNumber(int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            res = res ^ i ^ nums[i];
        }
        return res ^ nums.length;
    }

    public int nthUglyNumber(int n) {
        if (n <= 0) return 0;
        if (n == 1) return 1;
        int t2 = 0, t3 = 0, t5 = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            dp[i] = Math.min(dp[t2]*2, Math.min(dp[t3]*3, dp[t5]*5));
            if (dp[i] == dp[t2]*2) t2++;
            if (dp[i] == dp[t3]*3) t3++;
            if (dp[i] == dp[t5]*5) t5++;
        }
        return dp[n-1];
    }

    public boolean isUgly(int num) {
        if (num <= 0) return false;
        while (num % 2 == 0) num = num/2;
        while (num % 3 == 0) num = num/3;
        while (num % 5 == 0) num = num/5;
        return num == 1;
    }

    public int[] singleNumber(int[] nums) {
        int num = 0;
        for (int i : nums) {
            num = num ^ i;
        }
        int val = 1;
        while (true) {
            if ((num & val) != 0) {
                break;
            }
            val = val << 1;
        }
        int num1 = 0, num2 = 0;
        for (int i = 0; i < nums.length; i++) {
            if ((nums[i] & val) == 0) {
                num1 = num1 ^ nums[i];
            }
            else {
                num2 = num2 ^ nums[i];
            }
        }
        return new int[]{num1, num2};
    }

    public int addDigits(int num) {
        while (num >= 10) {
            int sum = 0;
            while (num != 0) {
                sum += num % 10;
                num = num / 10;
            }
            num = sum;
        }
        return num;
    }

    public List<String> binaryTreePaths(TreeNode root) {
        if (root == null) return new ArrayList<>();
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        getBinaryTreePaths(res, sb, root);
        return res;
    }

    private void getBinaryTreePaths(List<String> res, StringBuilder sb, TreeNode node) {
        if (node == null) return;
        int len = sb.length();
        sb.append(node.val);
        if (node.left == null && node.right == null) {
            res.add(sb.toString());
            sb.setLength(len);
            return;
        }

        sb.append("->");
        getBinaryTreePaths(res, sb, node.left);
        getBinaryTreePaths(res, sb, node.right);
        sb.setLength(len);
    }

    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        int[] array1 = new int[26];
        for (int i = 0; i < s.length(); i++) {
            array1[s.charAt(i) - 'a']++;
        }
        for (int i = 0; i < t.length(); i++) {
            array1[t.charAt(i) - 'a']--;
            if (array1[t.charAt(i) - 'a'] < 0) return false;
        }
        return true;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length < 1 || matrix[0].length < 1) {
            return false;
        }
        int col = matrix[0].length - 1;
        int row = 0;
        while (col >= 0 && row <= matrix.length - 1) {
            if (target == matrix[row][col])
                return true;
            else if (target < matrix[row][col])
                col--;
            else if (target > matrix[row][col])
                row++;
        }
        return false;
    }

    public int[] productExceptSelf(int[] nums) {
        if (nums == null) return null;
        if (nums.length == 0) return new int[0];

        int[] res = new int[nums.length];
        int mul = 1;
        int sum0 = 0;
        int loc0 = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                sum0++;
                loc0 = i;
            }
            else {
                mul *= nums[i];
            }
        }
        if (sum0 >= 2) {
            for (int i = 0; i < nums.length; i++) {
                res[i] = 0;
            }
        }
        else if (sum0 == 1) {
            for (int i = 0; i < nums.length; i++) {
                if (i == loc0) res[i] = mul;
                else res[i] = 0;
            }
        }
        else {
            for (int i = 0; i < nums.length; i++) {
                res[i] = mul / nums[i];
            }
        }
        return res;
    }


    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;
        if (root == p || root == q) return  root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) return root;
        else if (left != null) return left;
        else return right;
    }

    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val >= p.val && root.val <= q.val || root.val <= p.val && root.val >= q.val)
            return root;
        else if (root.val > p.val && root.val > q.val)
            return lowestCommonAncestor1(root.left, p, q);
        else return lowestCommonAncestor1(root.right, p, q);
    }


    public boolean isPalindrome(ListNode head) {
        if (head == null) return true;
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode mid = slow;

        slow = slow.next;
        ListNode pre;
        ListNode temp;
        pre = slow;
        while (slow != null) {
            if (mid.next != slow) {
                temp = slow;
                slow = slow.next;
                pre.next = temp.next;
                temp.next = mid.next;
                mid.next = temp;
            }
            else
                slow = slow.next;
        }

        ListNode node1 = mid.next;
        ListNode node2 = head;
        boolean res = true;
        while (node1 != null) {
            if (node1.val != node2.val) {
                res = false;
                break;
            }
            node1 = node1.next;
            node2 = node2.next;
        }

//        slow = mid.next;
//        pre = slow;
//        while (slow != null) {
//            if (mid.next != slow) {
//                temp = slow;
//                slow = slow.next;
//                pre.next = temp.next;
//                temp.next = mid.next;
//                mid.next = temp;
//            }
//            else
//                slow = slow.next;
//        }

        return res;
    }

    public int  countDigitOne(int n) {
        int res = 0;
        long factor = 1;
        long high = 0;
        long low = 0;
        long now = 0;
        if (n <= 0) return 0;
        while (n / factor != 0) {
            high = n / factor / 10;
            now = (n / factor) % 10;
            low = n - (n / factor) * factor;
            if (now == 0) {
                res += high * factor;
            }
            else if (now == 1) {
                res += high * factor + low + 1;
            }
            else {
                res += (high + 1) * factor;
            }
            factor = factor * 10;
        }
        return res;
    }


    static class MyQueue {

        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();

        /** Initialize your data structure here. */
        public MyQueue() {

        }

        /** Push element x to the back of queue. */
        public void push(int x) {
            if (stack2.empty())
                stack1.push(x);
            else {
                while (!stack2.empty()) {
                    stack1.push(stack2.pop());
                }
                stack1.push(x);
            }
        }

        /** Removes the element from in front of queue and returns that element. */
        public int pop() {
            if (stack1.empty()) {
                return stack2.pop();
            }
            else {
                while (!stack1.empty()) {
                    stack2.push(stack1.pop());
                }
                return stack2.pop();
            }
        }

        /** Get the front element. */
        public int peek() {
            if (stack1.empty()) {
                return stack2.pop();
            }
            else {
                while (!stack1.empty()) {
                    stack2.push(stack1.pop());
                }
                return stack2.peek();
            }
        }

        /** Returns whether the queue is empty. */
        public boolean empty() {
            return stack1.empty() && stack2.empty();
        }
    }

    public boolean isPowerOfTwo(int n) {
        int val = 1;
        for (int i = 0; i < 32; i++) {
            if ((val >> i) == n) return true;
        }
        return false;
    }

    public int kthSmallest(TreeNode root, int k) {
        int count = countBSTNodes(root.left);
        if (count >= k) return kthSmallest(root.left, k);
        else if (count + 1 > k) return kthSmallest(root.right, k - 1 - count);
        return root.val;
    }

    private int countBSTNodes(TreeNode root) {
        if (root == null) return 0;
        return 1 + countBSTNodes(root.left) + countBSTNodes(root.right);
    }

    public List<Integer> majorityElement(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0) return res;
        int count1 = 0, count2 = 0, value1 = nums[0], value2 = nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == value1) count1++;
            else if (nums[i] == value2) count2++;
            else if (count1 == 0) {
                value1 = nums[i];
                count1 = 1;
            }
            else if (count2 == 0) {
                value2 = nums[i];
                count2 = 1;
            }
            else {
                count1--;
                count2--;
            }
        }
        count1 = 0;
        count2 = 0;
        for (int i : nums) {
            if (i == value1) count1++;
            else if (i == value2) count2++;
        }
        if (count1 > nums.length / 3) res.add(value1);
        if (count2 > nums.length / 3) res.add(value2);
        return res;
    }

    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if (nums == null || nums.length == 0) return res;
        int start = nums[0], end = nums[0];
        for (int i = 1 ; i < nums.length; i++) {
            if (nums[i] - end == 1) {
                end = nums[i];
            }
            else {
                if (start == end) {
                    res.add(String.valueOf(start));
                }
                else {
                    res.add(new StringBuilder().append(start).append("->").append(end).toString());
                }
                start = nums[i];
                end = nums[i];
            }
        }
        if (start == end) {
            res.add(String.valueOf(start));
        }
        else {
            res.add(new StringBuilder().append(start).append("->").append(end).toString());
        }
        return res;
    }

    public int calculate(String s) {
        int len;
        if (s == null || (len = s.length()) == 0) return 0;
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        char sign = '+';
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (Character.isDigit(c)) {
                num = num * 10 + c - '0';
            }
            if (!(Character.isDigit(c)) && ' ' != c || i == len - 1) {
                if (sign == '-') stack.push(-num);
                if (sign == '+') stack.push(num);
                if (sign == '*') stack.push(stack.pop() * num);
                if (sign == '/') stack.push(stack.pop() / num);
                sign = s.charAt(i);
                num = 0;
            }
        }
        int re = 0;
        for (int i : stack) {
            re += i;
        }
        return re;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return root;
        }
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    class MyStack {

        Queue<Integer> queue;
        /** Initialize your data structure here. */
        public MyStack() {
            queue = new ArrayDeque<>();
        }

        /** Push element x onto stack. */
        public void push(int x) {
            queue.add(x);
            for (int i = 0; i < queue.size() - 1; i++)
                queue.offer(queue.poll());
        }

        /** Removes the element on top of the stack and returns that element. */
        public int pop() {
            return queue.poll();
        }

        /** Get the top element. */
        public int top() {
            return queue.peek();
        }

        /** Returns whether the stack is empty. */
        public boolean empty() {
            return queue.isEmpty();
        }
    }


    public int calculate1(String s) {
        Stack<Integer> stack = new Stack<Integer>();
        int result = 0;
        int number = 0;
        int sign = 1;
        for(int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if(Character.isDigit(c)){
                number = 10 * number + (int)(c - '0');
            }else if(c == '+'){
                result += sign * number;
                number = 0;
                sign = 1;
            }else if(c == '-'){
                result += sign * number;
                number = 0;
                sign = -1;
            }else if(c == '('){
                //we push the result first, then sign;
                stack.push(result);
                stack.push(sign);
                //reset the sign and result for the value in the parenthesis
                sign = 1;
                result = 0;
            }else if(c == ')'){
                result += sign * number;
                number = 0;
                result *= stack.pop();    //stack.pop() is the sign before the parenthesis
                result += stack.pop();   //stack.pop() now is the result calculated before the parenthesis

            }
        }
        if(number != 0) result += sign * number;
        return result;
    }

    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null) return 1;
        int height = 0;
        int nodeSum = 0;
        TreeNode curr = root;
        while (curr.left != null) {
            nodeSum += (1 << height);
            height++;
            curr = curr.left;
        }
        return nodeSum + countLastLevel(root, height);
    }

    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int left = Math.max(A, E), right = Math.max(Math.min(C, G), left);
        int bottom = Math.max(B, F), top = Math.max(Math.min(D,H), bottom);
        return (C - A) * (D - B) - (right - left) * (top - bottom) + (G - E) * (H - F);
    }

    private int countLastLevel(TreeNode root, int height) {
        if (height == 1) {
            if (root.right != null) return 2;
            else if (root.left != null) return 1;
            else return 0;
        }
        TreeNode midNode = root.left;
        int currHeight = 1;
        while (currHeight < height) {
            currHeight++;
            midNode = midNode.right;
        }
        if (midNode == null) return countLastLevel(root.left, height - 1);
        else return (1 << (height-1)) + countLastLevel(root.right, height - 1);
    }


    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (t < 0) return false;
        Map<Long, Long> d = new HashMap<>();
        long w = (long)t + 1;
        for (int i = 0; i < nums.length; ++i) {
            long m = getID(nums[i], w);
            if (d.containsKey(m)) return true;
            if (d.containsKey(m - 1) && Math.abs(nums[i] - d.get(m - 1)) < w)
                return true;
            if (d.containsKey(m + 1) && Math.abs(nums[i] - d.get(m + 1)) < w)
                return true;
            d.put(m, (long)nums[i]);
            if (i >= k) d.remove(getID(nums[i - k], w));
        }
        return false;
    }

    private long getID(long i, long w) {
        return i < 0 ? (i+1) / w - 1 : i / w;
    }

    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return 0;
        int row = matrix.length;
        int col = matrix[0].length;
        int[][] dp = new int[row + 1][col + 1];
        int res = 0;
        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++) {
                if (matrix[i-1][j-1] == '1') {
                    dp[i][j] = Math.min(dp[i-1][j], Math.min(dp[i-1][j-1], dp[i][j-1])) + 1;
                    res = Math.max(res, dp[i][j]);
                }
            }
        }
        return res * res;
    }

    public boolean containsNearbyDuplicate(int[] nums, int k) {
        int max = -1;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (nums[i] == nums[j]) {
                    max = Math.max(max, i-j);
                }
            }
        }
        return max == k;
    }

    public List<int[]> getSkyline(int[][] buildings) {
        List<int[]> res = new ArrayList<>();
        List<int[]> height = new ArrayList<>();
        for (int[] b:buildings) {
            height.add(new int[]{b[0], -b[2]});
            height.add(new int[]{b[1], b[2]});
        }
        Collections.sort(height, (a, b)->{
            if (a[0] != b[0])
                return a[0]-b[0];
            return a[1] - b[1];
        });

        Queue<Integer> pq = new PriorityQueue<>((a, b)-> (b - a));
        pq.offer(0);
        int prev = 0;
        for (int[] h:height){
            if (h[1] < 0) {
                pq.offer(-h[1]);
            } else {
                pq.remove(h[1]);
            }
            int cur = pq.peek();
            if (prev != cur) {
                res.add(new int[]{h[0],cur});
                prev = cur;
            }
        }
        return res;
    }

    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == nums[i+1]) return true;
        }
        return false;
    }

    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (n <= 0 || n >= 46) return res;
        getCombinationSum(res, new ArrayList<>(), 1, 0, k, n);
        return res;
    }

    private void getCombinationSum(List<List<Integer>> res, ArrayList<Integer> temp, int start, int sum, int k, int n) {
        if (sum > n) return;
        else if (sum == n && k == 0) {
            res.add((ArrayList<Integer>)temp.clone());
            return;
        }
        for (int i = start; i <= 9; i++) {
            temp.add(i);
            getCombinationSum(res, temp, i+1, sum+i, k - 1, n);
            temp.remove(temp.size() - 1);
        }
    }

    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0)
            return -1;
        return quickSelect(nums, 0, nums.length - 1, k - 1);
    }

    private int quickSelect(int[] nums, int start, int end, int k) {
        int left = start;
        int right = end;
        int pivot = nums[(end - start) / 2 + start];

        while (left <= right) {
            while (left <= right && nums[left] > pivot) {
                left++;
            }
            while (left <= right && nums[right] < pivot) {
                right--;
            }
            if (left <= right) {
                int swap = nums[left];
                nums[left] = nums[right];
                nums[right] = swap;
                left++;
                right--;
            }
        }

        if (start <= k && k <= right)
            return quickSelect(nums, start, right, k);
        else if (left <= k && k <= end)
            return quickSelect(nums, left, end, k);
        return nums[right+1];

    }

    public String shortestPalindrome(String s) {
        int j = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) == s.charAt(j))
                j += 1;
        }
        if (s.length() == j) return s;
        String suffix = s.substring(j);
        return new StringBuilder(suffix).reverse().toString() + shortestPalindrome(s.substring(0, j)) + suffix;
    }

    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        else if (nums.length == 2) return Math.max(nums[0], nums[1]);
        return Math.max(robStartAt(0, nums.length - 1, nums), robStartAt(1, nums.length, nums));
    }

    private int robStartAt(int start, int end, int[] nums) {
        int a = nums[start], b = Math.max(nums[start], nums[start +1]);
        int res = Math.max(a, b);
        for (int i =start+2; i < end; i++) {
            res = Math.max(a + nums[i], b);
            a = b;
            b = res;
        }
        return res;
    }

}
