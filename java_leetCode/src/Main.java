import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class Main {

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
        Main main = new Main();
        File f = new File("E:\\git\\starter.shop\\src\\main\\java\\cn\\sjwx\\starter\\shop");

    }



    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0) return;
        int rows = board.length;
        int cols = board[0].length;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                int liveCells = countLiveCell(board, i, j);
                int status = board[i][j];
                if (status == 1) {
                    if (liveCells < 2 || liveCells > 3) {
                        board[i][j] = 3;
                    }
                }
                else {
                    if (liveCells == 3)
                        board[i][j] = 4;
                }
            }

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == 3) {
                    board[i][j] = 0;
                }
                else if (board[i][j] == 4) {
                    board[i][j] = 1;
                }
            }
    }

    private int countLiveCell(int[][] board, int startX, int startY){
        int rows = board.length;
        int cols = board[0].length;

        int res = 0;
        for (int i = startX - 1; i<= startX + 1; i++) {
            for (int j = startY - 1; j <= startY + 1; j++) {
                if (i < 0 || i >= rows) continue;
                if (j < 0 || j >= cols) continue;
                if (i == startX && j == startY) continue;
                if (board[i][j] == 1 || board[i][j] == 3) res += 1;
            }
        }
        return res;
    }

    Map<String, List<Integer>> map1 = new HashMap<>();
    public List<Integer> diffWaysToCompute(String input) {
        if (map1.containsKey(input)) return map1.get(input);
        List<Integer> ret = new LinkedList<>();
        for (int i = 0; i<input.length(); i++) {
            if (input.charAt(i) == '-' ||
                    input.charAt(i) == '*' ||
                    input.charAt(i) == '+') {
                String part1 = input.substring(0, i);
                String part2 = input.substring(i + 1);
                List<Integer> part1Ret = diffWaysToCompute(part1);
                List<Integer> part2Ret = diffWaysToCompute(part2);

                for (Integer p1: part1Ret)
                    for (Integer p2 : part2Ret) {
                        int c = 0;
                        switch (input.charAt(i)) {
                            case '+':
                                c = p1 + p2;
                                break;
                            case '-':
                                c = p1 - p2;
                                break;
                            case '*':
                                c = p1 * p2;
                                break;
                        }
                        ret.add(c);
                    }
            }
        }
        if (ret.size() == 0) {
            ret.add(Integer.valueOf(input));
        }
        map1.put(input, ret);
        return ret;
    }

    static class TrieNode1 {
        TrieNode1[] children = new TrieNode1[26];
        String val;
    }

    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        TrieNode1 root = buildTrie(words);
        for (int i = 0; i < board.length; i++)
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, i, j, root, res);
            }
        return res;
    }

    private TrieNode1 buildTrie(String[] words) {
        TrieNode1 root = new TrieNode1();
        for (String w : words) {
            TrieNode1 p = root;
            for (char c : w.toCharArray()) {
                int i = c - 'a';
                if (p.children[i] == null) p.children[i] = new TrieNode1();
                p = p.children[i];
            }
            p.val = w;
        }
        return root;
    }


    private void dfs(char[][] board, int i, int j, TrieNode1 p, List<String> res) {
        char c = board[i][j];
        if (c == '#' || p.children[c - 'a'] == null) return;
        p = p.children[c - 'a'];
        if (p.val != null) {
            res.add(p.val);
            p.val = null;
        }

        board[i][j] = '#';
        if (i > 0) dfs(board, i-1, j, p, res);
        if (i < board.length - 1) dfs(board, i + 1, j, p, res);
        if (j > 0) dfs(board, i, j - 1, p, res);
        if (j < board[0].length - 1) dfs(board, i, j + 1, p, res);
        board[i][j] = c;
    }

    static class WordDictionary {
        static class WordNode {
            WordNode[] children = new WordNode[26];
            boolean isWord = false;
        }

        WordNode root;
        /** Initialize your data structure here. */
        public WordDictionary() {
            root = new WordNode();
        }

        /** Adds a word into the data structure. */
        public void addWord(String word) {
            WordNode node = root;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (node.children[c - 'a'] == null) {
                    node.children[c - 'a'] = new WordNode();
                }
                node = node.children[c - 'a'];
                if (i == word.length() - 1) node.isWord = true;
            }
        }

        /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
        public boolean search(String word) {
            return match(word.toCharArray(), 0, root);
        }

        private boolean match(char[] chars, int index, WordNode node) {
            if (index == chars.length) return node.isWord;
            else {
                char c = chars[index];
                if (c != '.') {
                    return node.children[c - 'a'] != null && match(chars, index+1, node.children[c - 'a']);
                }
                else {
                    for (int i = 0; i < node.children.length; i++) {
                        if (node.children[i] != null) {
                            if (match(chars, index+1, node.children[i]))
                                return true;
                        }
                    }
                    return false;
                }
            }
        }
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] deeps = new int[numCourses];
        List<List<Integer>> adjs = new ArrayList<>(numCourses);
        for (int i = 0; i < numCourses; i++) {
            adjs.add(new ArrayList<>());
        }
        for (int i = 0; i < prerequisites.length; i++) {
            int first = prerequisites[i][1];
            int second = prerequisites[i][0];
            deeps[second]++;
            adjs.get(first).add(second);
        }
        int[] order = new int[deeps.length];
        Queue<Integer> toVisit = new ArrayDeque<>();
        for (int i = 0; i < deeps.length; i++) {
            if (deeps[i] == 0) toVisit.offer(i);
        }
        int visited = 0;
        while (!toVisit.isEmpty()) {
            int from = toVisit.poll();
            order[visited++] = from;
            for (int to : adjs.get(from)) {
                deeps[to]--;
                if (deeps[to] == 0) toVisit.offer(to);
            }
        }
        return visited == deeps.length ? order : new int[0];
    }

    public int minSubArrayLen(int s, int[] nums) {
        if (nums.length == 0) return 0;
        int res = 0;
        int start = 0;
        int end = 0;
        int sum = nums[0];
        if (sum > s) return 1;
        for (int i =1; i < nums.length; i++){
            sum += nums[i];
            end++;
            if (sum < s) {
                continue;
            }
            else {
                while (start < end && sum >= s) {
                    if (sum - nums[start] >= s) {
                        sum = sum - nums[start];
                        start++;
                    }
                    else {
                        break;
                    }
                }
                if (res == 0)
                    res = end - start + 1;
                else {
                    if (end - start + 1 < res)
                        res = end - start + 1;
                }
            }
        }
        return res;
    }


    public int minSubArrayLen1(int s, int[] nums) {
        int length = nums.length;
        int slow = 0;
        int fast = 0;
        int sum = 0;
        int min = length + 1;
        while (fast < length) {
            while (sum < s && fast < length) {
                sum += nums[fast];
                fast++;
            }
            while (sum >= s) {
                sum -= nums[slow];
                slow ++;
            }
            min = Math.min(min, fast - slow + 1);
        }
        return min == length+1 ? 0 : min;
    }

    static class TrieNode{
        public char val;
        public boolean isWord;
        public TrieNode[] children = new TrieNode[26];
        public TrieNode(){}
        public TrieNode(char c) {
            TrieNode node = new TrieNode();
            node.val = c;
        }
    }

    static class Trie {
        private TrieNode root;

        /** Initialize your data structure here. */
        public Trie() {
            root = new TrieNode();
            root.val = ' ';
        }

        /** Inserts a word into the trie. */
        public void insert(String word) {
            TrieNode ws = root;
            for (int i =0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (ws.children[c - 'a'] == null) {
                    ws.children[c - 'a'] = new TrieNode(c);
                }
                ws = ws.children[c - 'a'];
            }
            ws.isWord = true;
        }

        /** Returns if the word is in the trie. */
        public boolean search(String word) {
            TrieNode ws = root;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (ws.children[c - 'a'] == null) return false;
                ws = ws.children[c - 'a'];
            }
            return ws.isWord;
        }

        /** Returns if there is any word in the trie that starts with the given prefix. */
        public boolean startsWith(String prefix) {
            TrieNode ws = root;
            for (int i = 0; i < prefix.length(); i++) {
                char c = prefix.charAt(i);
                if (ws.children[c - 'a'] == null) return false;
                ws = ws.children[c - 'a'];
            }
            return true;
        }
    }

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[][] matrix = new int[numCourses][numCourses];
        int[] indegree = new int[numCourses];

        for (int i =0 ; i < prerequisites.length; i++) {
            int ready = prerequisites[i][0];
            int pre = prerequisites[i][1];
            if (matrix[pre][ready] == 0)
                indegree[ready]++;
            matrix[pre][ready] = 1;
        }

        int count = 0;
        Queue<Integer> queue = new LinkedList<>();
        for (int i =0; i < indegree.length; i++) {
            if (indegree[i] == 0) queue.offer(i);
        }

        while (!queue.isEmpty()) {
            int course = queue.poll();
            count++;
            for (int i = 0; i < numCourses; i++){
                if (matrix[course][i] != 0)
                    if (--indegree[i] == 0)
                        queue.offer(i);
            }
        }
        return count == numCourses;
    }

    public ListNode reverseList(ListNode head) {
        if (head == null) return null;
        ListNode dummy = new ListNode(0);
        ListNode temp, node;
        while (head != null) {
            node = head;
            head = head.next;
            temp = dummy.next;
            dummy.next = node;
            node.next = temp;
        }
        return dummy.next;
    }

    public boolean isIsomorphic(String s, String t) {
        int[] m1 = new int[256], m2 = new int[256];
        int len = s.length();
        for (int i = 0; i < len; i++) {
            if (m1[s.charAt(i)] != m2[t.charAt(i)]) return false;
            m1[s.charAt(i)] = i + 1;
            m2[t.charAt(i)] = i + 1;
        }
        return true;
    }

    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int count = 0;
        for (int i = 2; i < n; i++){
            if (notPrime[i] == false){
                count++;
                for (int j = 2; i * j < n; j++){
                    notPrime[i*j] = true;
                }
            }
        }
        return count;
    }

    public ListNode removeElements(ListNode head, int val) {
        while (head != null && head.val == val) {
            head = head.next;
        }
        if (head == null) {
            return null;
        }
        ListNode prev = head;
        ListNode node = head.next;
        while (node != null) {
            if (node.val == val) {
                prev.next = node.next;
                node = node.next;
            } else {
                prev = node;
                node = node.next;
            }
        }
        return head;
    }

    public boolean isHappy(int n) {
        while (n > 1) {
            int newValue = 0;
            while (n != 0) {
                newValue += (n % 10) * (n % 10);
                n = n / 10;
            }
            n = newValue;
            if (n == 4) return false;
        }
        return true;
    }

    private int rangBitwiseAnd(int m, int n){
        if (m == 0) return 0;
        int moveFactor = 1;
        while (m != n) {
            m >>= 1;
            n >>= 1;
            moveFactor <<= 1;
        }
        return m * moveFactor;
    }

    private int numOfIsland = 0;
    public int numIslands(char[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    getIsland(grid, i, j);
                    numOfIsland ++;
                }
            }
        }
        return numOfIsland;
    }

    private void getIsland(char[][] grid, int i, int j) {
        if (grid[i][j] == '1') {
            grid[i][j] = '2';
        }
        if (i > 0) {
            getIsland(grid, i-1, j);
        }
        if (i < grid.length) {
            getIsland(grid, i+1, j);
        }
        if (j > 0) {
            getIsland(grid, i, j-1);
        }
        if (j < grid[0].length) {
            getIsland(grid, i, j+1);
        }
    }

    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) return new ArrayList<>();
        List<Integer> res = new ArrayList<>();
        getResOfRightSideView(res, 1, root);
        return res;
    }

    private void getResOfRightSideView(List<Integer> list, int deep, TreeNode node) {
        if (node == null) return;
        if (deep > list.size()) {
            list.add(node.val);
        }
        if (node.right != null) {
            getResOfRightSideView(list, deep + 1, node.right);
        }
        if (node.left != null)
            getResOfRightSideView(list, deep + 1, node.left);
    }

    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        else if (nums.length == 2) return Math.max(nums[0], nums[1]);
        int a = nums[0], b = Math.max(nums[0], nums[1]);
        int res = 0;
        for (int i =2; i < nums.length; i++) {
            res = Math.max(a + nums[i], b);
            a = b;
            b = res;
        }
        return res;
    }

    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            count += (n & 1);
            n = n >>>1;
        }
        return count;
    }

    public int reverseBits(int n) {
        int res = 0;
        for (int i=0; i < 32; i++) {
            res = (res << 1) + (n & 1);
            n >>>= 1;
        }
        return res;
    }

    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k-1);
        reverse(nums, k, nums.length - 1);
    }

    private void reverse(int[] nums, int k, int i) {
        while (k < i){
            int temp = nums[k];
            nums[k] = nums[i];
            nums[i] = temp;
            k++;
            i--;
        }
    }

    public List<String> findRepeatedDnaSequences(String s) {
        HashSet<String> first = new HashSet<>(), second = new HashSet<>();
        for (int i = 0; i + 9 < s.length(); i++) {
            String temp = s.substring(i, i+ 10);
            if (!first.add(temp))
                second.add(temp);
        }
        return new ArrayList<>(second);
    }

    public String largestNumber(int[] nums) {
        Comparator<String> comparator = new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String s1 = o1 + o2;
                String s2 = o2 + o1;
                return s2.compareTo(s1);
            }
        };
        String[] array = new String[nums.length];
        for (int i = 0; i < nums.length; i++){
            array[i] = String.valueOf(nums[i]);
        }

        Arrays.sort(array, comparator);
        if (array[0].charAt(0) == '0') return "0";
        StringBuilder sb = new StringBuilder();
        for (String i : array) {
            sb.append(i);
        }
        return sb.toString();
    }

    public int calculateMinimumHP(int[][] dungeon) {
        int row = dungeon.length;
        int col = dungeon[0].length;
        int[][] minHp = new int[row][col];
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                minHp[i][j] = 1;
        for (int i = row - 1; i >= 0; i--) {
            for (int j = col - 1; j >= 0; j--) {
                if ( i == row - 1 && j == col - 1) {
                    if (dungeon[i][j] > 0) {
                        minHp[i][j] = 1;
                    }
                    else {
                        minHp[i][j] = 1 - dungeon[i][j];
                    }
                }
                else if (i == row - 1) {
                    if (dungeon[i][j] > 0) {
                        minHp[i][j] = Math.max(minHp[i][j+1] - dungeon[i][j], 1);
                    }
                    else {
                        minHp[i][j] = minHp[i][j+1]  - dungeon[i][j];
                    }
                } else if (j == col - 1) {
                    if (dungeon[i][j] > 0) {
                        minHp[i][j] = Math.max(minHp[i+1][j] - dungeon[i][j], 1);
                    }
                    else {
                        minHp[i][j] = minHp[i+1][j]  - dungeon[i][j];
                    }
                } else {
                    if (dungeon[i][j] > 0) {
                        minHp[i][j] = Math.max(Math.min(minHp[i][j+1], minHp[i+1][j]) - dungeon[i][j], 1);
                    }
                    else {
                        minHp[i][j] = Math.min(minHp[i][j+1], minHp[i+1][j])  - dungeon[i][j];
                    }
                }
            }
        }
        return minHp[0][0];
    }

    public int trailingZeroes(int n) {
        int res = 0;
        while (n != 0) {
            res += (n / 5);
            n = n / 5;
        }
        return res;
    }

    public int titleToNumber(String s) {
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            res = res * 26 + c - 'A'+ 1;
        }
        return res;
    }

    public int majorityElement(int[] nums) {
        int res = 0;
        int count = 0;
        for (int i : nums){
            if (count == 0) {
                count++;
                res = i;
                continue;
            }
            if (i == res)
                count++;
            else count--;
        }
        return res;
    }

    public String convertToTitle(int n) {
        StringBuilder sb = new StringBuilder();
        int temp;
        while (n != 0) {
            temp = n % 26;
            if (temp == 0){
                n = (n / 26) - 1;
                sb.insert(0,'Z');
            }
            else {
                n = n / 26;
                sb.insert(0, (char) ('A' + temp - 1));
            }
        }
        return sb.toString();
    }


    public int[] twoSum(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while (left <= right) {
            int value = numbers[left] + numbers[right];
            if (value == target) {
                return new int[]{left+1, right+1};
            }
            else if (value > target) right--;
            else left++;
        }
        return new int[2];
    }


    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        // "+" or "-"
        res.append(((numerator > 0) ^ (denominator > 0)) ? "-" : "");
        long num = Math.abs((long)numerator);
        long den = Math.abs((long)denominator);

        // integral part
        res.append(num / den);
        num %= den;
        if (num == 0) {
            return res.toString();
        }

        // fractional part
        res.append(".");
        HashMap<Long, Integer> map = new HashMap<Long, Integer>();
        map.put(num, res.length());
        while (num != 0) {
            num *= 10;
            res.append(num / den);
            num %= den;
            if (map.containsKey(num)) {
                int index = map.get(num);
                res.insert(index, "(");
                res.append(")");
                break;
            }
            else {
                map.put(num, res.length());
            }
        }
        return res.toString();
    }

    public int compareVersion(String version1, String version2) {
        int val1, val2 ;
        int len1 = version1.length(), len2 = version2.length();
        int idx1 = 0, idx2 = 0;
        while (idx1 < len1 || idx2 < len2) {
            val1 = 0;
            val2 = 0;
            while (idx1 < len1 && version1.charAt(idx1) != '.') {
                val1 = val1 * 10 + version1.charAt(idx1) - '0';
                idx1++;
            }
            while (idx2 < len2 && version2.charAt(idx2) != '.') {
                val2 = val2 * 10 + version2.charAt(idx2) - '0';
                idx2++;
            }
            if (val1 > val2) return 1;
            else if (val1 < val2) return -1;
            else {
                idx1++;
                idx2++;
            }
        }
        return 0;
    }

    public int maximumGap(int[] nums) {
        if (nums.length < 2) return 0;

        int max = nums[0];
        int min = nums[0];
        for (int i = 1; i < nums.length; i++) {
            max = Math.max(max, nums[i]);
            min = Math.min(min, nums[i]);
        }
        int buckets = (int)Math.ceil((double)(max - min) / (nums.length - 1));
        int[] MAX = new int[nums.length - 1];
        int[] MIN = new int[nums.length - 1];

        Arrays.fill(MAX, Integer.MIN_VALUE);
        Arrays.fill(MIN, Integer.MAX_VALUE);

        for (int i : nums) {
            if (i == max || i == min) continue;
            int idx = (i - min) / buckets;
            MAX[idx] = Math.max(MAX[idx], i);
            MIN[idx] = Math.min(MIN[idx], i);
        }

        int pre = min;
        int maxGap = 0;
        for (int i = 0; i < MAX.length; i ++) {
            if(MAX[i] == Integer.MIN_VALUE) continue;
            else {
                if (MIN[i] - pre > maxGap) {
                    maxGap = MIN[i] - pre;
                }
                pre = MAX[i];
            }
        }
        if (max - pre > maxGap) {
            maxGap = max - pre;
        }
        return maxGap;
    }

    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            if (left == right) {
                return left;
            }
            else if (left + 1 == right) {
                if (nums[left] > nums[right]) return left;
                else return right;
            }
            else {
                int mid = (left + right) / 2;
                if (nums[mid] > nums[mid - 1]) {
                    if (nums[mid] > nums[mid + 1]) return mid;
                    else left = mid + 1;
                }
                else if (nums[mid] < nums[mid - 1]) {
                    right = mid - 1;
                }
            }
        }
        return 0;
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a = headA;
        ListNode b = headB;
        while (a != null && b != null) {
            if (a == b) return a;
            a = a.next;
            b = b.next;
        }

        ListNode A = headA;
        ListNode B = headB;

        if (a != null) {
            while (a != null) {
                A = A.next;
                a = a.next;
            }
        }
        else if (b != null) {
            while (b != null) {
                B = B.next;
                b = b.next;
            }
        }

        while (A != null && B != null) {
            if (A == B) return A;
            A = A.next;
            B = B.next;
        }
        return null;
    }


    static class MinStack {

        Stack<Integer> valStack;
        Stack<Integer> minStack;
        /** initialize your data structure here. */
        public MinStack() {
            valStack = new Stack<>();
            minStack = new Stack<>();
        }

        public void push(int x) {
            valStack.push(x);
            if (minStack.empty())
                minStack.push(x);
            else  {
                int min = minStack.peek();
                if (min >= x) {
                    minStack.push(x);
                }
            }
        }

        public void pop() {
            int i = valStack.pop();
            if (i == minStack.peek())
                minStack.pop();
        }

        public int top() {
            return valStack.peek();
        }

        public int getMin() {
            return minStack.peek();
        }
    }

    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            }
            else if (nums[mid] < nums[left]){
                right = mid;
            }
            else {
                right--;
            }
        }
        return nums[left];
    }

    public int maxProduct(int[] nums) {
        int r = nums[0];
        int n = nums.length;
        for (int i = 1, imax = r, imin = r; i < n; i++) {
            if (nums[i] < 0) {
                int temp = imax;
                imax = imin;
                imin = temp;
            }
            imax = Math.max(nums[i], nums[i] * imax);
            imin = Math.min(nums[i], nums[i] * imin);
            r = Math.max(r, imax);
        }
        return r;
    }

    public int findMin1(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            if (nums[left] <= nums[right])
                return nums[left];
            else {
                int mid = (left + right) / 2;
                if (nums[mid] >= nums[left]) {
                    left = mid + 1;
                }
                else {
                    right = mid;
                }
            }
        }
        return nums[0];
    }

    public String reverseWords(String s) {
        if(s == null || s.trim().length() == 0) return "";
        StringBuilder sb = new StringBuilder();
        String[] array = s.trim().split(" ");
        for (int i = array.length - 1; i >=1; i--) {
            if (array[i].trim().length() == 0) continue;
            sb.append(array[i]).append(" ");
        }
        if(array[0].trim().length() != 0)
            sb.append(array[0]);
        return sb.toString();
    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        int s1, s2;
        int len = tokens.length;
        for (int i = 0; i < len; i++) {
            String s = tokens[i];
            if (s.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            }
            else if (s.equals("-")) {
                s2 = stack.pop();
                s1 = stack.pop();
                stack.push(s1 - s2);
            }
            else if (s.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            }
            else if (s.equals("/")){
                s2 = stack.pop();
                s1 = stack.pop();
                stack.push(s1 / s2);
            }
            else stack.push(Integer.parseInt(s));
        }
        return stack.pop();
    }

    public int maxPoints(Point[] points) {
        if (points == null) return 0;
        int len = points.length;
        if (len <= 2) return len;

        Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
        int res = 0;
        for (int i = 0; i < len; i++) {
            map.clear();
            int overlap = 0, max = 0;
            for (int j = i+1; j < len; j++) {
                int x = points[i].x - points[j].x;
                int y = points[i].y - points[j].y;
                if (x == 0 && y == 0) {
                    overlap++;
                    continue;
                }
                int gcd = gcd(x, y);
                if (gcd != 0) {
                    x /= gcd;
                    y /= gcd;
                }
                if (map.containsKey(x)) {
                    if (map.get(x).containsKey(y)) {
                        map.get(x).put(y, map.get(x).get(y) +1);
                    }
                    else {
                        map.get(x).put(y,1);
                    }
                } else {
                    Map<Integer, Integer> m = new HashMap<>();
                    m.put(y,1);
                    map.put(x, m);
                }
                max = Math.max(max, map.get(x).get(y));
            }
            res = Math.max(res, max + overlap + 1);
        }
        return res;
    }

    private int gcd(int a, int b) {
        if (b==0) return a;
        else return gcd(b,a%b);
    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode fast, slow, pre = null;
        fast = slow = head;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        pre.next = null;
        ListNode l1 = sortList(head);
        ListNode l2 = sortList(slow);
        return merge(l1, l2);
    }

    private ListNode merge(ListNode list1, ListNode list2) {
        ListNode pre = new ListNode(0);
        ListNode it1 = list1, it2 = list2, head = pre;
        while (it1 != null && it2 != null) {
            if (it1.val > it2.val) {
                head.next = it2;
                head = head.next;
                it2 = it2.next;
            }
            else {
                head.next = it1;
                head = head.next;
                it1 = it1.next;
            }
        }
        if (it1 != null) {
            head.next = it1;
        }
        if (it2 != null) {
            head.next = it2;
        }
        return pre.next;
    }

    public ListNode insertionSortList(ListNode head) {
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode it = head;
        ListNode temp, innerIt;
        while (it.next != null) {
            if (it.val > it.next.val) {
                temp = it.next;
                it.next = temp.next;
                innerIt = pre;
                while (innerIt.next.val < temp.val) {
                    innerIt = innerIt.next;
                }
                temp.next = innerIt.next;
                innerIt.next = temp;
            }
            else {
                it = it.next;
            }
        }
        return pre.next;
    }




    static class LRUCache {
        class DLinkedNode {
            int key;
            int value;
            DLinkedNode pre;
            DLinkedNode post;
        }

        private void addNode(DLinkedNode node){
            node.pre = head;
            node.post = head.post;

            head.post.pre = node;
            head.post = node;
        }

        private void removeNode(DLinkedNode node){
            DLinkedNode pre = node.pre;
            DLinkedNode post = node.post;

            pre.post = post;
            post.pre = pre;
        }

        private void moveToHead(DLinkedNode node){
            this.removeNode(node);
            this.addNode(node);
        }

        private DLinkedNode popTail(){
            DLinkedNode res = tail.pre;
            this.removeNode(res);
            return res;
        }

        HashMap<Integer, DLinkedNode> map = new HashMap<>();
        private int count;
        private int capacity;
        private DLinkedNode head, tail;

        public LRUCache(int capacity) {
            this.count = 0;
            this.capacity = capacity;

            head = new DLinkedNode();
            head.pre = null;

            tail = new DLinkedNode();
            tail.post = null;

            head.post = tail;
            tail.pre = head;
        }

        public int get(int key) {

            DLinkedNode node = map.get(key);
            if(node == null){
                return -1; // should raise exception here.
            }

            // move the accessed node to the head;
            this.moveToHead(node);

            return node.value;
        }


        public void put(int key, int value) {
            DLinkedNode node = map.get(key);

            if(node == null){

                DLinkedNode newNode = new DLinkedNode();
                newNode.key = key;
                newNode.value = value;

                this.map.put(key, newNode);
                this.addNode(newNode);

                ++count;

                if(count > capacity){
                    // pop the tail
                    DLinkedNode tail = this.popTail();
                    this.map.remove(tail.key);
                    --count;
                }
            }else{
                // update the value.
                node.value = value;
                this.moveToHead(node);
            }

        }
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        LinkedList<Integer> res = new LinkedList<>();
        Deque<TreeNode> deque = new ArrayDeque<>();
        while (!deque.isEmpty() || root != null) {
            if (root != null) {
                deque.push(root);
                res.addFirst(root.val);
                root = root.right;
            }
            else {
                root = deque.pop().left;
            }
        }
        return res;
    }

    public List<Integer> postorderTraversal1(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        getPostorderTraversal(root, res);
        return res;
    }

    private void getPostorderTraversal(TreeNode root, List<Integer> res) {
        if (root == null) return;
        getPostorderTraversal(root.right, res);
        getPostorderTraversal(root.left, res);
        res.add(root.val);
    }

    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (root != null) {
            res.add(root.val);
            if (root.right != null)
                stack.push(root.right);
            root = root.left;
            if (root == null && !stack.isEmpty())
            {
                root = stack.pop();
            }
        }
        return res;
    }

    public List<Integer> preorderTraversal1(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        getPreorderTraversal(root, res);
        return res;
    }

    private void getPreorderTraversal(TreeNode root, List<Integer> res) {
        if (root == null) return;
        res.add(root.val);
        getPreorderTraversal(root.left, res);
        getPreorderTraversal(root.right, res);
    }

    public void reorderList(ListNode head) {
        ArrayList<ListNode> list = new ArrayList<>();
        ListNode node = head;
        while (node != null) {
            list.add(node);
            node = node.next;
        }
        int len = list.size();
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode temp = pre;
        for (int i = 0; i < len / 2; i++) {
            ListNode first = list.get(i);
            ListNode last = list.get(len - 1 - i);
            temp.next = first;
            first.next = last;
            temp = last;
        }
        if (len % 2 == 0) {
            temp.next = null;
        }
        else {
            temp.next = list.get(len / 2);
            temp.next.next = null;
        }
        return;
    }

    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;
        boolean isCycle = false;
        if (fast == null || fast.next == null) return null;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                isCycle = true;
                break;
            }
        }
        if (!isCycle) return null;
        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }

    public boolean hasCycle(ListNode head) {
        ListNode fast = head, slow = head;
        if (fast == null || fast.next == null) return false;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) return true;
        }
        return false;
    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        Set<String> dict = new HashSet<>(wordDict);
        return dfs(s, dict, new HashMap<>());
    }

    private List<String> dfs(String s, Set<String> dict, HashMap<String, LinkedList<String>> map) {
        if (map.containsKey(s)) return map.get(s);

        LinkedList<String> res = new LinkedList<>();
        if(s.length() == 0) {
            res.add("");
            return res;
        }
        for (String word : dict) {
            if (s.startsWith(word)) {
                List<String> subList = dfs(s.substring(word.length()), dict, map);
                for (String sub : subList) {
                    res.add(word + (sub.isEmpty() ? "" : " ") + sub);
                }
            }
        }
        map.put(s, res);
        return res;
    }

    public boolean wordBreak1(String s, List<String> wordDict) {
        if (s == null || s.length() == 0) return false;
        Set<String> set = new HashSet<>(wordDict);
        int len = s.length();
        boolean[] dp = new boolean[len + 1];
        dp[0] = true;
        for (int i=1; i <= len; i++) {
            for (int j=0; j<i; j++) {
                String sub = s.substring(j, i);
                if (set.contains(sub) && dp[j]) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[len];
    }

    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) return null;
        Map<RandomListNode, RandomListNode> randomListMap = new HashMap<>();
        RandomListNode node = head;
        while (node != null) {
            randomListMap.put(node, new RandomListNode(node.label));
            node = node.next;
        }
        node = head;
        while (node != null) {
            randomListMap.get(node).next = randomListMap.get(node.next);
            randomListMap.get(node).random = randomListMap.get(node.random);
            node = node.next;
        }
        return randomListMap.get(head);
    }


    public int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        int len = nums.length;
        for (int i=0; i<len; i++) {
            ones = (ones ^ nums[i]) & ~twos;
            twos = (twos ^ nums[i]) & ~ones;
        }
        return ones;
    }

    public int singleNumber2(int[] nums) {
        int res = 0;
        int len = nums.length;
        for (int i=0; i<len; i++) {
            res ^= nums[i];
        }
        return res;
    }

    public int candy(int[] ratings) {
        int len = ratings.length;
        if (len <= 1) return len;
        int[] num = new int[len];
        num[0] = 1;
        for (int i = 1; i < len ; i++) {
            num[i] = 1;
            if (ratings[i] > ratings[i-1])
                num[i] = num[i-1] + 1;
        }
        for (int i = len -1; i > 0; i--)
            if (ratings[i-1] > ratings[i])
                num[i-1] = Math.max(num[i] + 1, num[i-1]);
        int res = 0;
        for (int i = 0; i < len; i++) {
            res+=num[i];
        }
        return res;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int len = gas.length;
        int[] minus = new int[len];
        for (int i = 0; i < len; i++)
            minus[i] = gas[i] - cost[i];
        int remain = 0;
        int start = 0;
        for (int i = 0; i < len; i++) {
            remain += minus[i];
            if (remain < 0) {
                remain = 0;
                start = i+1;
                if (start == len)
                    return -1;
            }
        }
        int res = start;
        remain = 0;
        for (int i = 0; i < len; i++) {
            if (start + i == len)
                start = -i;
            remain += minus[start + i];
            if (remain < 0)
                return -1;
        }
        return res;
    }

    private HashMap<Integer, UndirectedGraphNode> map = new HashMap<>();
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        return clone(node);
    }

    private UndirectedGraphNode clone(UndirectedGraphNode node) {
        if (node == null) return node;
        if (map.containsKey(node.label)) return map.get(node.label);

        UndirectedGraphNode clone = new UndirectedGraphNode(node.label);
        map.put(clone.label, clone);
        for (UndirectedGraphNode neighbor : node.neighbors) {
            clone.neighbors.add(clone(neighbor));
        }
        return clone;
    }

    private List<List<String>> resList;
    private ArrayList<String> curList;
    public List<List<String>> partition(String s) {
        resList = new ArrayList<>();
        curList = new ArrayList<>();
        backTrack(s, 0);
        return resList;
    }

    private void backTrack(String s, int l) {
        if (curList.size() > 0 && l >= s.length()) {
            List<String> r = (ArrayList<String>) curList.clone();
            resList.add(r);
        }
        for (int i = l; i < s.length(); i++) {
            if (isPalindrome(s, l, i)) {
                if (i == l) {
                    curList.add(Character.toString(s.charAt(i)));
                }
                else
                    curList.add(s.substring(l, i+1));
                backTrack(s, i+1);
                curList.remove(curList.size() -1);
            }
        }
    }

    private boolean isPalindrome(String s, int l, int r) {
        if (l == r) return true;
        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) return false;
            l++;
            r--;
        }
        return true;
    }

    public void solve(char[][] board) {
        int row = board.length;
        if (row == 0) return;
        int col = board[0].length;

        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++) {
                if (i == 0) {
                    solveBoard(board, i, j, row -1, col -1);
                }
                if (i == row-1) {
                    solveBoard(board, i, j, row-1, col-1);
                }
                if (j == 0) {
                    solveBoard(board, i, j, row -1, col -1);
                }
                if (j == col-1) {
                    solveBoard(board, i, j, row-1, col-1);
                }
            }

        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++) {
                if (board[i][j] == 'O')
                    board[i][j] = 'X';
            }

        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++) {
                if (board[i][j] == '1')
                    board[i][j] = 'O';
            }

    }

    private void solveBoard(char[][] board, int i, int j, int row, int col) {
        if (board[i][j] == 'O') {
            board[i][j] = '1';
            if (i > 0) {
                solveBoard(board, i-1, j, row, col);
            }
            if (i < row) {
                solveBoard(board, i+1, j, row, col);
            }
            if (j > 0) {
                solveBoard(board, i, j-1, row, col);
            }
            if (j < col) {
                solveBoard(board, i, j + 1, row, col);
            }
        }
    }

    int addValue = 0;
    public int sumNumbers(TreeNode root) {
        if (root == null) return 0;
        getSumNumbers(root, 0);
        return addValue;
    }

    private void getSumNumbers(TreeNode root, int tempValue) {
        tempValue = tempValue * 10 + root.val;
        if (root.left == null && root.right == null)
            addValue += tempValue;
        if (root.left != null)
            getSumNumbers(root.left, tempValue);
        if (root.right != null)
            getSumNumbers(root.right, tempValue);
    }

    public int ladderLength(String start, String end, Set<String> dict) {
        Queue<String> queue = new LinkedList<>();
        queue.add(start);
        queue.add(null);

        Set<String> visited = new HashSet<>();
        visited.add(start);

        int level = 1;
        while (!queue.isEmpty()) {
            String str = queue.poll();

            if (str != null) {
                for (int i = 0; i < str.length(); i++) {
                    char[] chars = str.toCharArray();
                    for (char c = 'a'; c <= 'z'; c++) {
                        chars[i] = c;
                        String word = new String(chars);

                        if (word.equals(end)) return level+1;

                        if (dict.contains(word) && !visited.contains(word)) {
                            queue.add(word);
                            visited.add(word);
                        }
                    }
                }
            }
            else {
                level++;
                if (!queue.isEmpty()) {
                    queue.add(null);
                }
            }
        }
        return 0;
    }

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        HashSet<String> dict = new HashSet<>(wordList);
        List<List<String>> res = new ArrayList<>();
        HashMap<String, ArrayList<String>> nodeNeighbors = new HashMap<>();
        HashMap<String, Integer> distance = new HashMap<>();
        ArrayList<String> solution = new ArrayList<>();
        dict.add(beginWord);
        bfs(beginWord, endWord, dict, nodeNeighbors, distance);
        dfs(beginWord, endWord, dict, nodeNeighbors, distance, solution, res);
        return res;
    }

    private void bfs(String beginWord, String endWord, HashSet<String> dict, HashMap<String, ArrayList<String>> nodeNeighbors, HashMap<String, Integer> distance) {
        for (String str : dict)
            nodeNeighbors.put(str, new ArrayList<>());
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);
        distance.put(beginWord, 0);

        while (!queue.isEmpty()) {
            int count = queue.size();
            boolean foundEnd = false;
            for (int i = 0; i < count; i++) {
                String cur = queue.poll();
                int curDistance = distance.get(cur);
                ArrayList<String> neighbors = getNeighbors(cur, dict);
                for (String neighbor : neighbors) {
                    nodeNeighbors.get(cur).add(neighbor);
                    if (!distance.containsKey(neighbor)) {
                        distance.put(neighbor, curDistance+1);
                        if (endWord.equals(neighbor))
                            foundEnd = true;
                        else
                            queue.offer(neighbor);
                    }
                }
            }

            if (foundEnd)
                break;
        }
    }

    private ArrayList<String> getNeighbors(String cur, HashSet<String> dict) {
        ArrayList<String> res = new ArrayList<>();
        char chs[] = cur.toCharArray();

        for (char ch = 'a'; ch <= 'z'; ch++) {
            for (int i = 0; i < chs.length; i++) {
                if (chs[i] == ch) continue;
                char old_ch = chs[i];
                chs[i] = ch;
                if (dict.contains(String.valueOf(chs)))
                    res.add(String.valueOf(chs));
                chs[i] = old_ch;
            }
        }
        return res;
    }

    private void dfs(String beginWord, String endWord, HashSet<String> dict, HashMap<String, ArrayList<String>> nodeNeighbors, HashMap<String, Integer> distance, ArrayList<String> solution, List<List<String>> res) {
        solution.add(beginWord);
        if (endWord.equals(beginWord))
            res.add(new ArrayList<>(solution));
        else {
            for (String next : nodeNeighbors.get(beginWord))
                if (distance.get(next) == distance.get(beginWord) + 1)
                    dfs(next, endWord, dict, nodeNeighbors, distance, solution, res);
        }
        solution.remove(solution.size() - 1);
    }

    public int longestConsecutive(int[] nums) {
        if (nums.length == 0) return 0;
        int res = 1;
        Arrays.sort(nums);
        int temp = 1;
        for (int i = 1; i < nums.length ; i++) {
            if (nums[i] - nums[i-1] == 1) temp++;
            else if (nums[i] == nums[i-1]) continue;
            else {
                if (temp > res)
                    res = temp;
                temp = 1;
            }
        }
        if (temp > res)
            res = temp;
        return res;
    }

    public boolean isPalindrome(String s) {
        List<Character> list = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if ( isAlpha(c) || isNumeric(c))
                list.add(c);
        }
        int len = list.size();
        if (len <= 1) return true;
        for (int i = 0 ; i <= list.size() / 2; i++) {
            char first = list.get(i);
            char last = list.get(list.size() - 1 - i);
            if (first == last || (Math.abs(first - last) == 32 && isAlpha(first) && isAlpha(last) ))
                continue;
            else
                return false;
        }
        return true;
    }

    private static boolean isAlpha(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    private static boolean isNumeric(char c) {
        return (c >= '0' && c <= '9');
    }

    int maxValue;
    public int maxPathSum(TreeNode root) {
        maxValue = Integer.MIN_VALUE;
        getMaxPathSum(root);
        return maxValue;
    }

    private int getMaxPathSum(TreeNode root) {
        if (root == null) return 0;
        int left = Math.max(0, getMaxPathSum(root.left));
        int right = Math.max(0, getMaxPathSum(root.right));
        maxValue = Math.max(maxValue, left + root.val + right);
        return Math.max(left, right) + root.val;
    }


    public int maxProfit2(int[] prices) {
        int hold1 = Integer.MIN_VALUE, hold2 = Integer.MIN_VALUE;
        int release1 = 0, release2 = 0;
        for(int curPrice:prices){                              // Assume we only have 0 money at first
            release2 = Math.max(release2, hold2+curPrice);     // The maximum if we've just sold 2nd stock so far.
            hold2    = Math.max(hold2,    release1-curPrice);  // The maximum if we've just buy  2nd stock so far.
            release1 = Math.max(release1, hold1+curPrice);     // The maximum if we've just sold 1nd stock so far.
            hold1    = Math.max(hold1,    -curPrice);          // The maximum if we've just buy  1st stock so far.
        }
        return release2; ///Since release1 is initiated as 0, so release2 will always higher than release1.
    }

    public int maxProfit1(int[] prices) {
        if (prices.length == 0) return 0;
        int res = 0;
        int temp = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] >= temp) {
                res += prices[i] - temp;
                temp = prices[i];
            }
            else{
                temp = prices[i];
            }
        }
        return res;
    }

    public int maxProfit(int[] prices) {
        if (prices.length <= 1)
            return 0;
        int max = 0;
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < prices.length; i++) {
            min = Math.min(min, prices[i]);
            max = Math.max(max, prices[i] - min);
        }
        return max;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int[] dp = new int[triangle.size()];
        for (int i = 0; i < triangle.size(); i++) {
            List<Integer> temp = triangle.get(i);
            for (int j = temp.size() - 1; j >= 0; j--) {
                if(temp.size() == 1)
                    dp[j] += temp.get(j);
                else{
                    if (j == temp.size() - 1){
                        dp[j] = temp.get(j) + dp[j-1];
                    }
                    else if(j < temp.size() - 1 && j > 0)
                        dp[j] = Math.min(dp[j], dp[j-1]) + temp.get(j);
                    else if (j == 0)
                        dp[j] += temp.get(j);
                }
            }
        }

        int res = dp[0];
        for (int val : dp)
            if (val < res)
                res = val;
        return res;
    }

    public List<Integer> getRow(int rowIndex) {
        List<Integer> list = new ArrayList<>();
        if (rowIndex >= 1) {
            list.add(1);
        }
        while (--rowIndex > 0) {
            for (int i = list.size() - 1;  i >= 1 ; i--) {
                int val = list.get(i);
                list.set(i, val + list.get(i - 1));
            }
            list.add(1);
        }
        return list;
    }


    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        if (numRows >= 1) {
            list.add(1);
            res.add(new ArrayList<>(list));
        }
        while (--numRows > 0) {
            for (int i = list.size() - 1;  i >= 1 ; i--) {
                int val = list.get(i);
                list.set(i, val + list.get(i - 1));
            }
            list.add(1);
            res.add(new ArrayList<>(list));
        }
        return res;
    }

    public class TreeLinkNode {
        TreeLinkNode left;
        TreeLinkNode right;
        TreeLinkNode next;
    }

    public void connect(TreeLinkNode root) {
        if(root == null) return;
        LinkedList<TreeLinkNode> list = new LinkedList<>();
        list.addLast(root);
        list.addLast(null);
        while (!list.isEmpty()) {
            TreeLinkNode node = list.removeFirst();
            if (node != null) {
                TreeLinkNode temp = list.getFirst();
                node.next = temp;
                if (node.left != null) list.addLast(node.left);
                if (node.right != null) list.addLast(node.right);
            }
            else {
                if (list.size() != 0)
                    list.addLast(null);
            }
        }
    }

    public int numDistinct(String s, String t){
        int[][] dp = new int[s.length()+1][t.length()+1];
        for (int i = 0; i < dp.length; i++){
            dp[i][0] = 1;
        }
        for (int i = 1; i <= s.length(); i++){
            for (int j = 1; j <= t.length(); j++) {
                if (s.charAt(i-1) == t.charAt(j-1))
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
                else
                    dp[i][j] = dp[i-1][j];
            }
        }
        return dp[s.length()][t.length()];
    }

    private TreeNode prev = null;

    public void flatten(TreeNode root) {
        if (root == null)
            return;
        flatten(root.right);
        flatten(root.left);
        root.right = prev;
        root.left = null;
        prev = root;
    }

    public TreeNode sortedListToBST(ListNode head) {
        return toBST(head, null);
    }

    private TreeNode toBST(ListNode head, ListNode tail) {
        ListNode slow = head;
        ListNode fast = head;
        if (head == tail) return null;
        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }

        TreeNode root = new TreeNode(slow.val);
        root.left = toBST(head, slow);
        root.right = toBST(slow.next, tail);
        return root;
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> reverseRes = new ArrayList<>();
        getRes(reverseRes, root, 0);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = reverseRes.size() - 1; i >= 0; i--) {
            res.add(reverseRes.get(i));
        }
        return res;
    }

    private void getRes(List<List<Integer>> reverseRes, TreeNode root, int deep) {
        if (root == null)
            return;
        List<Integer> temp;
        if (reverseRes.size() <= deep) {
            temp = new ArrayList<>();
            reverseRes.add(temp);
        }
        else {
            temp = reverseRes.get(deep);
        }
        temp.add(root.val);
        getRes(reverseRes, root.left, deep+1);
        getRes(reverseRes, root.right, deep+1);
    }

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if(inorder == null || postorder == null || inorder.length != postorder.length)
            return null;
        return helper(inorder, 0, inorder.length - 1, postorder, 0 , postorder.length - 1);
    }

    public TreeNode helper(int[] inorder, int inStart, int inEnd, int[] postorder, int postStart, int postEnd) {
        if (postStart > postEnd || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val)
                index = i;
        }
        root.left = helper(inorder, inStart, index - 1, postorder, postStart, postStart + index - inStart - 1);
        root.right = helper(inorder, index + 1, inEnd, postorder, postStart + index - inStart, postEnd - 1);
        return root;
    }

    public TreeNode buildTree1(int[] preorder, int[] inorder) {
        return helper1(0, 0, inorder.length - 1, preorder, inorder);
    }

    public TreeNode helper1(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
        if (preStart > preorder.length - 1 || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val)
                index = i;
        }
        root.left = helper1(preStart + 1, inStart, index - 1, preorder, inorder);
        root.right = helper1(preStart + index - inStart + 1, index + 1, inEnd , preorder, inorder);
        return root;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> sol = new ArrayList<>();
        travel(root, sol, 0);
        return sol;
    }

    private void travel(TreeNode curr, List<List<Integer>> sol, int level) {
        if (curr == null) return;
        if (sol.size() <= level) {
            List<Integer> newLevel = new LinkedList<>();
            sol.add(newLevel);
        }

        List<Integer> collection = sol.get(level);
        if (level % 2 == 0) collection.add(curr.val);
        else collection.add(0, curr.val);

        travel(curr.left, sol, level + 1);
        travel(curr.right, sol, level + 1);
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null)
            return s2 == null ? s3 == null : s2.equals(s3);
        if (s2 == null)
            return s1 == null ? s3 == null : s1.equals(s3);
        int len1 = s1.length();
        int len2 = s2.length();
        if (s3.length() != len1 + len2)
            return false;
        boolean[][] matrix = new boolean[len1 + 1][len2 + 1];
        for (int i = 0; i <= len1; i++){
            for (int j = 0; j <= len2; j++){
                if (i == 0 && j == 0) matrix[i][j] = true;
                else if (i == 0) matrix[i][j] = matrix[i][j - 1] && s2.charAt(j - 1) == s3.charAt(j - 1);
                else if (j == 0) matrix[i][j] = matrix[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i - 1);
                else matrix[i][j] = (matrix[i-1][j] && s1.charAt(i) == s3.charAt(i+j-1)) || (matrix[i][j-1] && s2.charAt(j-1) == s3.charAt(i+j-1));
            }
        }
        return matrix[len1][len2];
    }

    public List<TreeNode> generateTrees(int n) {
        if (n <= 0) return new ArrayList<>();
        return genTrees(1, n);
    }

    private List<TreeNode> genTrees(int left, int right) {
        List<TreeNode> list = new ArrayList<>();

        if (left > right){
            list.add(null);
            return list;
        }

        List<TreeNode> lNodes, rNodes;
        for (int i = left; i <= right; i++) {
            lNodes = genTrees(left, i - 1);
            rNodes = genTrees(i + 1, right);

            for (TreeNode lNode : lNodes) {
                for (TreeNode rNode : rNodes) {
                    TreeNode node = new TreeNode(i);
                    node.left = lNode;
                    node.right = rNode;
                    list.add(node);
                }
            }
        }

        return list;
    }

    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        if (s == null || s.length() < 3 || s.length() > 12) {
            return res;
        }
        getIp(res, new ArrayList<String>(), s,0, 0);
        return res;
    }

    private void getIp(List<String> res, ArrayList<String> strings, String s, int deep, int start) {
        if (deep == 4) {
            if (start == s.length()) {
                StringBuilder sb = new StringBuilder();
                for (String s1 : strings){
                    sb.append(s1).append('.');
                }
                sb.deleteCharAt(sb.length() - 1);
                res.add(sb.toString());
            }
            return;
        }
        for (int i = 1; i <= 3; i++){
            if (start + i > s.length())
                return;
            String temp = s.substring(start, start + i);
            if (temp.length() == 1 || (!temp.startsWith("0") && Integer.valueOf(temp) < 256)) {
                strings.add(temp);
                getIp(res, strings, s, deep + 1, start + i);
                strings.remove(strings.size() - 1);
            }
        }
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (m == n) return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode pre = dummy, loopPre, temp;
        for (int i = 0; i < m - 1; i++) {
            pre = pre.next;
        }
        loopPre = pre.next;
        for (int i = 0; i < n - m; i++){
            temp = loopPre.next;
            loopPre.next = temp.next;
            temp.next = pre.next;
            pre.next = temp;
        }
        return dummy.next;
    }

    public int numDecodings(String s) {
        int len = s.length();
        if (len == 0)
            return 0;
        int[] array = new int[len+1];
        array[len] = 1;
        array[len - 1] = s.charAt(len - 1) == '0' ? 0 : 1;
        for (int i = len - 2; i >= 0; i--) {
            if (s.charAt(i) == '0') continue;
            else if (Integer.parseInt(s.substring(i, i+2)) > 26) {
                array[i] = array[i + 1];
            }
            else {
                array[i] = array[i + 1] + array[i + 2];
            }
        }
        return array[0];
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i <= nums.length; i++) {
            getSubSetsWithDup(res, new ArrayList<>(), nums, 0, i);
        }
        return res;
    }

    private void getSubSetsWithDup(List<List<Integer>> res, List<Integer> temp, int[] nums, int start, int deep){
        if (deep == 0) {
            List<Integer> list = new ArrayList<>(temp);
            res.add(list);
            return;
        }
        int i = start;
        while (i <= nums.length - 1){
            temp.add(nums[i]);
            getSubSetsWithDup(res, temp, nums, i+1, deep - 1);
            temp.remove(temp.size() - 1);
            i++;
            while (i <= nums.length - 1 && nums[i] == nums[i - 1]) i++;
        }
    }

    public List<Integer> grayCode(int n) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < 1<<n; i++) result.add(i^(i>>1));
        return result;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m + n - 1;
        m--;
        n--;
        while (m >= 0 && n >= 0){
            if (nums1[m] > nums2[n])
                nums1[i--] = nums1[m--];
            else
                nums1[i--] = nums2[n--];
        }
        while (n >= 0){
            nums1[i--] = nums2[n--];
        }
    }


    public boolean isScramble(String s1, String s2) {
        if (s1.equals(s2)) return true;
        int[] letters = new int[26];
        for (int i = 0; i<s1.length(); i++) {
            letters[s1.charAt(i) - 'a']++;
            letters[s2.charAt(i) - 'a']--;
        }
        for (int i =0; i < 26; i++)
            if (letters[i] != 0) return false;

        for (int i=1; i < s1.length(); i++) {
            if (isScramble(s1.substring(0,i), s2.substring(0,i))
                    && isScramble(s1.substring(i), s2.substring(i))) return true;
            if (isScramble(s1.substring(0,i), s2.substring(s2.length() - i))
                    && isScramble(s1.substring(i), s2.substring(0, s2.length() - i))) return true;
        }
        return false;
    }

    public ListNode partition(ListNode head, int x) {
        if (head == null) return null;
        ListNode mid = new ListNode(x);
        ListNode pre = null, res = null;
        ListNode after = null;
        List<ListNode> list = new ArrayList<>();
        for (ListNode temp = head; temp != null; temp = temp.next) {
            if (temp.val > x) {
                if (after == null) {
                    after = temp;
                    mid.next = after;
                }
                else {
                    after.next = temp;
                    after = after.next;
                }
            }
            else if (temp.val < x) {
                if (pre == null) {
                    pre = temp;
                    pre.next = mid;
                    res = pre;
                }
                else {
                    pre.next = temp;
                    pre = pre.next;
                }
            }
            else {
                list.add(temp);
            }
        }
        if (after != null)
            after.next = null;

        for (ListNode node : list) {
            if (pre == null) {
                pre = node;
                res = pre;
            }
            else {
                pre.next = node;
                pre = pre.next;
            }
        }
        if (res == null)
            return mid.next;
        else {
            pre.next = mid.next;
            return res;
        }
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null)
            return null;
        ListNode res = head;
        ListNode pre = res;
        head = head.next;
        while (head != null) {
            if (head.val != pre.val) {
                pre.next = head;
                pre = pre.next;
            }
            head = head.next;
        }
        pre.next = null;
        return res;
    }

    public ListNode deleteDuplicates2(ListNode head) {
        if (head == null)
            return null;
        ListNode res = new ListNode(0);
        res.next = head;
        ListNode pre = res;
        while (head.next != null) {
            if (head.next.val == head.val) {
                while (head.next != null && head.next.val == head.val) {
                    head = head.next;
                }
                pre.next = head;
                head = head.next;
                if (head == null) return res.next;
            }
            else {
                pre = pre.next;
                head = head.next;
            }
        }
        return res.next;
    }

    public boolean search(int[] nums, int target) {
        int right = nums.length - 1, mid, left = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] == target) return true;

            if (nums[left] == nums[mid] && nums[right] == nums[left] ) {
                ++left;
                --right;
            }
            else if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && nums[mid] > target) right = mid - 1;
                else left = mid + 1;
            }
            else {
                if (nums[mid] < target && nums[right] >= target) left = mid + 1;
                else right = mid - 1;
            }
        }
        return false;
    }

    public int removeDuplicates(int[] nums) {
        int i = 0;
        for (int n : nums) {
            if (i < 2 || n > nums[i - 2])
                nums[i++] = n;
        }
        return i;
    }

    public boolean exist(char[][] board, String word) {
        if (word == null || word.isEmpty())
            return false;
        char firstChar = word.charAt(0);
        for (int i = 0; i < board.length ; i++)
            for (int j = 0; j < board[0].length ; j++)
                if (board[i][j] == firstChar) {
                    boolean ok = existWord(board, word, i, j);
                    if (ok) return true;
                }
        return false;
    }

    private boolean result = false;

    private boolean existWord(char[][] board, String word, int startX, int startY) {
        boolean[][] path = new boolean[board.length][board[0].length];
        for (int i = 0; i < path.length; i++)
            for (int j = 0; j < path[0].length; j++)
                path[i][j] = true;
        path[startX][startY] = false;
        existWord(board, path, word, startX, startY, 1);
        return result;
    }

    private void existWord(char[][] board, boolean[][] path, String word, int startX, int startY, int matchPos) {
        if (result == true)
            return ;
        if (matchPos == word.length()) {
            result = true;
            return ;
        }
        char c = word.charAt(matchPos);
        if (startX > 0 && board[startX - 1][startY] == c && path[startX - 1][startY] == true) {
            path[startX - 1][startY] = false;
            existWord(board, path, word, startX - 1, startY, matchPos + 1);
            path[startX - 1][startY] = true;
        }
        if (startX < board.length - 1 && board[startX + 1][startY] == c && path[startX + 1][startY] == true) {
            path[startX + 1][startY] = false;
            existWord(board, path, word, startX + 1, startY, matchPos + 1);
            path[startX + 1][startY] = true;
        }
        if (startY > 0 && board[startX][startY - 1] == c && path[startX][startY - 1] == true) {
            path[startX][startY - 1] = false;
            existWord( board, path, word, startX, startY - 1, matchPos + 1);
            path[startX][startY - 1] = true;
        }
        if (startY < board[0].length - 1 && board[startX][startY + 1] == c && path[startX][startY + 1] == true) {
            path[startX][startY + 1] = false;
            existWord(board, path, word, startX, startY + 1, matchPos + 1);
            path[startX][startY + 1] = true;
        }
    }


    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i <= nums.length; i++) {
            getSubSets(res, new ArrayList<>(), nums, 0, i);
        }
        return res;
    }

    private void getSubSets(List<List<Integer>> res, List<Integer> temp, int[] nums, int start, int deep){
        if (deep == 0) {
            List<Integer> list = new ArrayList<>(temp);
            res.add(list);
            return;
        }
        for (int i = start; i <= nums.length - 1; i++) {
            temp.add(nums[i]);
            getSubSets(res, temp, nums, i+1, deep - 1);
            temp.remove(temp.size() - 1);
        }
    }

    public List<List<Integer>> combine(int n, int k) {
        if (k > n) return new ArrayList<>();
        List<List<Integer>> res = new ArrayList<>();
        getCombination(res, new ArrayList<>(), n, 1, k);
        return res;
    }

    private void getCombination(List<List<Integer>> res, List<Integer> temp, int n, int start, int deep){
        if (deep == 0) {
            List<Integer> list = new ArrayList<>(temp);
            res.add(list);
            return;
        }
        for (int i = start; i <= n; i++) {
            temp.add(i);
            getCombination(res, temp, n, i+1, deep - 1);
            temp.remove(temp.size() - 1);
        }
    }

    public void sortColors(int[] nums) {
        int count[] = new int[3];
        for (int i =0; i < nums.length; i++)
            count[nums[i]]++;
        int i = 0;
        int temp = 0;
        while (i <= 2) {
            if (count[i] == 0){
                i++;
            }
            else {
                nums[temp++] = i;
                count[i]--;
            }
        }
    }


    public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix.length == 0 || matrix[0].length == 0)
            return false;
        int left = 0;
        int col = matrix.length;
        int row = matrix[0].length;
        int right = col * row - 1;
        while (left <= right) {
            int mid = ( left + right) / 2;
            int x = mid/ row, y = (mid % row);
            if (matrix[x][y] > target)
                right = mid - 1;
            else if (matrix[x][y] < target)
                left = mid + 1;
            else
                return true;
        }
        return false;
    }


    public void setZeroes(int[][] matrix) {
        int col0 = 1, col = matrix.length, row = matrix[0].length;
        for (int i = 0; i < col; i++) {
            if (matrix[i][0] == 0) col0 = 0;
            for (int j = 1; j < row; j++) {
                if (matrix[i][j] == 0)
                {
                    matrix[0][j] = matrix[i][0] = 0;
                }
            }
        }
        for (int i = 1; i < row; i++) {
            if (matrix[0][i] == 0) {
                for (int j = 1; j < col; j++) {
                    matrix[j][i] = 0;
                }
            }
        }
        for (int i = 0; i < col; i++) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < row; j++) {
                    matrix[i][j] = 0;
                }
            }
        }

        if (col0 == 0) {
            for (int i = 0; i < col; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    public int minDistance(String word1, String word2) {
        int len1 = word1.length(), len2 = word2.length();
        if (len1 == 0) return len2;
        if (len2 == 0) return len1;
        int[][] matrix = new int[len1 + 1][len2 + 1];
        for (int i = 0; i <= len1; i++)
            matrix[i][0] = i;
        for (int i = 0; i <= len2; i++)
            matrix[0][i] = i;
        int i = 0, j = 0;
        for (i = 1; i <= len1; i++)
        {
            for (j = 1; j <= len2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1))
                    matrix[i][j] = matrix[i-1][j-1];
                else matrix[i][j] = Math.min(matrix[i-1][j-1]+1, Math.min(matrix[i-1][j] + 1, matrix[i][j-1] + 1));
            }
        }
        return matrix[len1][len2];
    }

    public String simplifyPath(String path) {
        path = path.replaceAll("//", "/");
        String[] strings = path.split("/");
        List<String> list = new LinkedList<>();
        for (int i = 0; i < strings.length; i ++) {
            if (!"".equals(strings[i]) )
                list.add(strings[i]);
        }
        for (int i = 0; i < list.size(); i ++) {
            String temp = list.get(i);
            if (temp.equals(".")) {
                list.remove(i);
                i--;
            }
            else if (temp.equals("..")) {
                list.remove(i);
                i--;
                if (i >= 0)
                {
                    list.remove(i);
                    i--;
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++){
            sb.append("/").append(list.get(i));
        }
        if (sb.length() == 0)
            sb.append("/");
        return sb.toString();
    }

    public BigInteger climbStairs(int n) {
        if (n <= 0) new BigInteger("0");
        if (n == 1) new BigInteger("1");
        if (n == 2) new BigInteger("2");

        BigInteger f1 = new BigInteger("1"), f2 = new BigInteger("2");
        for (int i = 2; i < n; i++) {
            f2 = f1.add(f2);
            f1 = f2.subtract(f1);
        }
        return f2;
    }

    public int mySqrt(int x) {
        if(x == 0)
            return 0;
        int left = 1;
        int right = x;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (mid > x / mid )
                right = mid - 1;
            else if (mid < x / mid)
                left = mid + 1;
            else {
                return mid;
            }
        }
        return left < right ? left : right;
    }

    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < words.length; i ++) {
            if (sb.length() == 0) {
                sb.append(words[i]);
                continue;
            }
            int len1 = sb.length();
            int len2 = words[i].length();
            if (len1 + 1 + len2 > maxWidth) {
                res.add(addAdjustString(sb.toString(), maxWidth));
                sb = new StringBuilder();
            }
            if (sb.length() != 0)
                sb.append(" ");
            sb.append(words[i]);
        }
        if (sb.length() != 0)
        {
            for (int i = sb.length(); i < maxWidth; i++)
                sb.append(" ");
            res.add(sb.toString());
        }
        return res;
    }

    private String addAdjustString(String s, int maxWidth) {
        String[] strs = s.split(" ");
        StringBuilder sb = new StringBuilder();
        if(strs.length == 1){
            sb.append(strs[0]);
            for (int i = sb.length(); i < maxWidth; i++)
                sb.append(" ");
            return sb.toString();
        }

        int spaces = strs.length - 1;
        int len = s.length() - spaces;
        int spaceCount = (maxWidth - len) / spaces;
        int maxSpaces = (maxWidth - len) % spaces;

        int count = 0;
        for (int i = 0; i < strs.length; i++) {
            sb.append(strs[i]);
            if (i == strs.length - 1)
                break;
            for (int j = 0; j < spaceCount; j++) {
                sb.append(" ");
            }
            if (count < maxSpaces) {
                count++;
                sb.append(" ");
            }
        }
        return sb.toString();
    }

    public String addBinary(String a, String b) {
        int aSize = a.length();
        int bSize = b.length();
        char[] res = new char[aSize > bSize ? aSize + 1 : bSize + 1];
        int loc = res.length - 1;
        int carryBit = 0;
        char aTemp, bTemp;
        for (int i = 0; i < res.length; i++) {
            if (i >= aSize)
                aTemp = '0';
            else
                aTemp = a.charAt(aSize - i - 1);
            if (i >= bSize)
                bTemp = '0';
            else
                bTemp = b.charAt(bSize - i - 1);
            res[loc] = (char)('0' + ((aTemp - '0' + bTemp - '0' + carryBit) % 2));
            carryBit = (aTemp - '0' + bTemp - '0' + carryBit) / 2;
            loc --;
        }
        if (res[0] == '0')
            return new String(res, 1, res.length - 1);
        return new String(res);
    }

    public int[] plusOne(int[] digits) {
        int carrybit = 0;
        int[] res;
        digits[digits.length - 1]++;
        for (int i = digits.length - 1; i >= 0; i--) {
            digits[i] += carrybit;
            if (digits[i] == 10) {
                digits[i] = 0;
                carrybit = 1;
            }
            else
                break;
        }
        if (carrybit == 1){
            res = new int[digits.length + 1];
            res[0] = 1;
            res[1] = 0;
            for (int i = 1; i < digits.length; i++) {
                res[i+1] = digits[i];
            }
        }
        else {
            res = digits;
        }
        return res;
    }

    public void getAllJavaFile(String path) {
        File root = new File(path);
        List<File> res = new ArrayList<>();
        getAllJavaFile(res, root);
        int count = 0;
        int max = 0;
        String fileName = "";
        for (int i =0; i < res.size(); i++) {
            int line = getLine(res.get(i));
            count += line;
            if (line > max) {
                max = line;
                fileName = res.get(i).getName();
            }
        }
        System.out.println(max);
        System.out.println(fileName);
        System.out.println(res.size());
        System.out.println(count);
    }

    private int getLine(File f) {
        FileReader fileReader = null;
        LineNumberReader lineNumberReader = null;
        int res = 0;
        try {
            fileReader = new FileReader(f);
            lineNumberReader = new LineNumberReader(fileReader);
            while (lineNumberReader.readLine() != null) {
                res++;
            }
            return res;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                lineNumberReader.close();
                fileReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return  res;
    }

    public void getAllJavaFile(List<File> res, File base) {
        if (base.exists() && base.isFile() && base.getName().endsWith(".java")) {
            res.add(base);
        }
        else if (base.isDirectory()) {
            File[] files = base.listFiles();
            for (int i = 0; i < files.length; i++) {
                getAllJavaFile(res, files[i]);
            }
        }
    }

    private int res;

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {

        int row = obstacleGrid.length;
        int col = obstacleGrid[0].length;
        for (int i = 0; i < row; i++) {
            for (int j =0; j < col; j ++) {
                if (obstacleGrid[i][j] == 1) {
                    obstacleGrid[i][j] = 0;
                }
                else if (obstacleGrid[i][j] == 0) {
                    if (i == 0 && j == 0)
                        obstacleGrid[i][j] = 1;
                    else if (j == 0 && i != 0)
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j];
                    else if (j != 0 && i == 0)
                        obstacleGrid[i][j] = obstacleGrid[i][j - 1];
                    else {
                        obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
                    }
                }
            }
        }
        return obstacleGrid[row - 1][col - 1];
    }

    public int uniquePaths(int m, int n) {
        m = m - 1;
        n = n - 1;
        m = m + n;
        n = n < m - n ? n : m - n;
        long res = 1;
        for (int i = 1; i <= n; i++) {
            res = res * m;
            m--;
        }
        for (int i = 1; i <= n; i++) {
            res = res / i;
        }
        return (int)res;
    }

    public List<List<String>> solveNQueens2(int n) {

        List<List<String>> res = new ArrayList<>();
        getAns(res, new ArrayList<Integer>(), 0, n);
        return res;
    }

    private void getAns(List<List<String>> res, List<Integer> temp, int deep, int n) {
        if (!isOk(temp, deep, n)) {
            return;
        }
        if (deep == n) {
            res.add(construct(temp, n));
            return;
        }
        for (int i = 0; i < n; i++) {
            temp.add(i);
            getAns(res, temp, deep + 1, n);
            temp.remove(temp.size() - 1);
        }
    }

    private boolean isOk(List<Integer> temp, int deep, int n) {
        deep = deep - 1;
        if (deep == 0 || deep == -1)
            return true;
        int now = temp.get(deep);
        for (int i = 0; i < deep; i++) {
            int iTemp = temp.get(i);
            if (now == iTemp || Math.abs(now - iTemp) == Math.abs(deep - i))
                return false;
        }
        return true;
    }

    private List<String> construct(List<Integer> temp, int n){
        List<String> res = new ArrayList<>();
        for (int i = 0; i < temp.size(); i ++) {
            int iTemp = temp.get(i);
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j <= iTemp - 1; j++)
                sb.append('.');
            sb.append('Q');
            for (int j = iTemp; j < n-1; j++)
                sb.append('.');
            res.add(sb.toString());
        }
        return res;
    }


    public int[][] generateMatrix(int n) {
        int matrix[][] = new int[n][n];
        if (n == 0)
            return matrix;
        int count = 1;
        int left = 0;
        int right = n - 1;
        int top = 0;
        int bottom = n - 1;
        while (true) {
            for (int i = left; i <= right; i++) {
                matrix[top][i] = count++;
            }
            top++;
            if (top > bottom)
                break;
            for (int i = top; i <= bottom; i++) {
                matrix[i][right] = count++;
            }
            right--;
            if (left > right)
                break;
            for (int i = right; i >= left; i--) {
                matrix[bottom][i] = count++;
            }
            bottom--;
            if (top > bottom)
                break;
            for (int i = bottom; i >= top; i--) {
                matrix[i][left] = count++;
            }
            left++;
            if (left > right)
                break;
        }
        return matrix;
    }

    public int lengthOfLastWord(String s) {
        if (s == null || s.isEmpty())
            return 0;
        String[] words = s.split(" ");
        return words[words.length - 1].length();
    }

    class Interval {
        int start;
        int end;
        Interval() { start = 0; end = 0; }
        Interval(int s, int e) { start = s; end = e; }
    }

    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        if (newInterval == null)
            return intervals;
        List<Interval> res = new ArrayList<>();
        if (intervals == null || intervals.size() == 0) {
            res.add(newInterval);
            return res;
        }
        Interval now = new Interval();
        now.start = newInterval.start;
        now.end = newInterval.end;
        Interval temp;
        for (int i = 0; i < intervals.size(); i++) {
            temp = intervals.get(i);
            if ((now.start <= temp.start && now.end >= temp.start) || (now.start <= temp.end && now.end >= temp.end)) {
                now.start = now.start < temp.start ? now.start : temp.start;
                now.end = now.end > temp.end ? now.end : temp.end;
            }
            else {
                res.add(temp);
            }
        }
        res.add(now);
        res.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start > o2.start ? 1 : o1.start == o2.start ? 0 : -1;
            }
        });
        return res;
    }

    public List<Interval> merge(List<Interval> intervals) {
        if (intervals == null || intervals.size() == 0)
            return new ArrayList<>();
        Collections.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start > o2.start ? 1 : o1.start == o2.start ? 0 : -1;
            }
        });
        List<Interval> res = new ArrayList<>();
        Interval now = intervals.get(0);
        Interval temp;
        for (int i = 1; i < intervals.size(); i++) {
            temp = intervals.get(i);
            if (now.end < temp.start) {
                res.add(now);
                now = temp;
            }
            else if (temp.end > now.end){
                now.end = temp.end;
            }
        }
        res.add(now);
        return res;
    }

    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0)
            return false;
        int temp = 0;
        int max = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i == max && nums[i] == 0)
                return false;
            temp = i + nums[i];
            if (temp > max)
                max = temp;
        }
        return true;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
            return res;
        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;
        while (true) {
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }
            top++;
            if (top > bottom)
                break;
            for (int i = top; i <= bottom; i++) {
                res.add(matrix[i][right]);
            }
            right--;
            if (left > right)
                break;
            for (int i = right; i >= left; i--) {
                res.add(matrix[bottom][i]);
            }
            bottom--;
            if (top > bottom)
                break;
            for (int i = bottom; i >= top; i--) {
                res.add(matrix[i][left]);
            }
            left++;
            if (left > right)
                break;
        }
        return res;
    }

    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int res = nums[0];
        int temp = res;
        for (int i = 1; i < nums.length; i++) {
            temp += nums[i];
            if (temp < 0 || temp < res)
                temp = 0;
            if (temp > res)
                res = temp;
        }
        return res;
    }

    public List<List<String>> solveNQueens(int n) {
        char[][] matrix = new char[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i][j] = '.';

        List<List<String>> res = new ArrayList<>();
        getAns(res, matrix, 0);
        return res;
    }

    private void getAns(List<List<String>> res, char[][] matrix, int deep) {
        if (!isOk(matrix, deep)) {
            return;
        }
        if (deep == matrix.length )
        {
            res.add(construct(matrix));
            return;
        }
        for (int i = 0 ; i < matrix.length; i++) {
            matrix[deep][i] = 'Q';
            getAns(res, matrix, deep + 1);
            matrix[deep][i] = '.';
        }
    }

    private boolean isOk(char[][] matrix, int deep) {
        if (deep == 0)
            return true;
        int loc = 0;
        for (int j = 0 ; j < matrix.length; j++) {
            if (matrix[deep - 1][j] == 'Q')
            {
                loc = j;
                break;
            }
        }
        for (int j = 0 ; j < matrix.length; j++) {
            if (matrix[deep - 1][j] == 'Q')
            {
                loc = j;
                break;
            }
        }

        for (int i = 0; i < deep - 1 ; i ++) {
            int temp = 0;
            for (int j = 0; j < matrix.length ; j++) {
                if (matrix[i][j] == 'Q')
                {
                    temp = j;
                    if (temp == loc || ((temp - loc) == (deep - 1 - i))  || ((temp - loc) == -(deep - 1- i)))
                        return false;
                    else
                        break;
                }
            }
        }
        return true;
    }

    private List<String> construct(char[][] matrix) {
        List<String> res = new ArrayList<>();

        for (int i = 0; i < matrix.length ; i++) {
            String temp = new String(matrix[i]);
            res.add(temp);
        }
        return res;
    }

    public static double myPow(double x, int n) {
        if (n == Integer.MIN_VALUE) {
            return myPow(x, n + 1) / x;
        }
        else if (n < 0) {
            n = -n;
            x = 1.0 / x;
        }

        double res = 1.0;
        while (n != 0) {
            if (n % 2 == 1) {
                res = res * x;
            }
            x = x * x;
            n = n >> 1;
        }
        return res;
    }

    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (int i = 0; i < strs.length; i++) {
            char[] chars = strs[i].toCharArray();
            Arrays.sort(chars, 0 ,chars.length);
            String temp = new String(chars);
            if (map.containsKey(temp)) {
                List<String> set = map.get(temp);
                set.add(strs[i]);
                map.put(temp, set);
            }
            else {
                List<String> set = new ArrayList<>();
                set.add(strs[i]);
                map.put(temp, set);
            }
        }
        return new ArrayList<List<String>>(map.values());
    }

    public static void rotate(int[][] matrix) {
        int len1 = matrix.length;
        int len2 = matrix[0].length;
        for (int i = 0; i < len1; i++) {
            for (int j = i+1; j < len2 ; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        for (int i = 0; i < len1; i++)
            for (int j = 0; j < len2 / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][len2 - 1 -j];
                matrix[i][len2 - 1 -j] = temp;
            }
    }

    public static List<List<Integer>> permute(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        res.add(getList(nums));
        while (getNextPermute(nums)) {
            res.add(getList(nums));
        }
        return res;
    }

    private static boolean getNextPermute(int[] nums) {
        int i;
        for (i = nums.length - 1; i >= 1; i--) {
            if (nums[i] > nums[i - 1]) {
                int k = nums[i - 1];
                int j = i;
                while ( j < nums.length - 1 && nums[j] > k)
                    j++;

                if (nums[j] <= k)
                    j--;
                swap(nums, i- 1, j);
                Arrays.sort(nums, i, nums.length);
                return true;
            }
        }
        return false;
    }

    private static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private static List<Integer> getList(int[] nums) {
        List<Integer> list = new ArrayList<>();
        for (int i =0; i<nums.length; i++) {
            list.add(nums[i]);
        }
        return list;
    }


    public static int jump(int[] nums) {
        int len = nums.length;
        int count = 0;
        int max = 0;
        int lastMax = 0;
        for (int i = 0; i < len - 1; i++) {
            if (max < nums[i] + i) {
                max = nums[i] + i;
            }
            if (i == lastMax) {
                lastMax = max;
                count ++;
                if (lastMax >= len - 1)
                    return count;
            }
        }
        return count;
    }

    public static boolean isMatch(String s, String p) {
        boolean[][] match = new boolean[s.length()+ 1][p.length() + 1];
        match[s.length()][p.length()] = true;
        for (int i = p.length() - 1; i>= 0;i--) {
            if (p.charAt(i) != '*')
                break;
            else
                match[s.length()][i] = true;
        }

        for (int i = s.length()-1; i >= 0; i--)
            for (int j = p.length() - 1; j >= 0; j--)
                if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')
                    match[i][j] = match[i+1][j+1];
                else if (p.charAt(j) == '*')
                    match[i][j] = match[i+1][j] || match[i][j+1];
                else
                    match[i][j] = false;

        return match[0][0];
    }


    public static String multiply(String num1, String num2) {
        int len1 = num1.length();
        int len2 = num2.length();
        int[] ans = new int[len1 + len2];

        for (int i = num1.length() - 1; i >= 0; i--) {
            for (int j = num2.length() - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int s = len1 + len2 - (len1 + len2 - i - j - 1);
                if (mul >= 10) {
                    ans[s] += mul % 10;
                    adjust(ans, s);
                    ans[s - 1] += mul / 10;
                    adjust(ans, s - 1);
                }
                else {
                    ans[s] += mul;
                    adjust(ans, s);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        int start = 0;
        for (int i=0; i<ans.length; i++)
            if (ans[i] != 0) {
                start = i;
                break;
            }
        for (int i = start; i < ans.length; i++) {
            sb.append(ans[i]);
        }
        return sb.toString();
    }

    private static void adjust(int[] ans, int i) {
        if (i > ans.length || i <= 0)
            return;
        if (ans[i] < 10)
            return;
        else {
            ans[i] -= 10;
            ans[i-1]++;
            adjust(ans, i-1);
        }
    }
}
