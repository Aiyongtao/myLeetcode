import java.io.File;
import java.util.*;

public class Twice {

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

    public static double sqrt(double c){
        if(c < 0) return Double.NaN; //既然要开平方，肯定不能为负啊
        double err = 1e-8; //精度
        double x = c; //迭代的初始值
        while(Math.abs(x - c/x) > err){ //没达到精度，那么继续迭代
            x = (x + c/x) / 2.0;
        }
        return x;
    }


    public static void main(String[] args) {
        "".indexOf("");
        System.out.println(sqrt(100));
    }

    public void nextPermutation(int[] nums) {
    }


    public int strStr(String haystack, String needle) {

        if (needle.equals("")) return 0;
        if (haystack.equals("")) return -1;
        char[] arr = needle.toCharArray();
        int[] next = getNext(arr);
        for (int i = 0, j = 0, end = haystack.length(); i < end;) {
            if (j == -1 || haystack.charAt(i) == arr[j]) {
                j++;
                i++;
                if (j == arr.length) return i - arr.length;
            }
            if (i < end && haystack.charAt(i) != arr[j]) j = next[j];
        }
        return -1;
    }

    private int[] getNext(char[] arr) {
        int len = arr.length;
        int[] next = new int[len];
        next[0] = -1;
        for (int i =0, j = -1; i + 1 < len;) {
            if (j == -1 || arr[i] == arr[j]) {
                next[i + 1] = j + 1;
                if (arr[i+1] == arr[j+1]) next[i+1] = next[j + 1];
                i++;
                j++;
            }
            if (arr[i] != arr[j]) j = next[j];
        }
        return next;
    }

    public int removeElement(int[] nums, int val) {
        if (nums == null || nums.length == 0) return 0;
        int slow = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != val)
                nums[slow++] = nums[i];
        }
        return slow;
    }

    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int slow = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1])
                nums[slow++] = nums[i];
        }
        return slow;
    }


    public ListNode swapPairs(ListNode head) {
        if (head == null) return null;
        ListNode dummy = new ListNode(0);
        ListNode temp = dummy;
        temp.next = head;
        ListNode first;
        ListNode second;
        while (temp.next != null && temp.next.next != null) {
            first = temp.next;
            second = temp.next.next;
            temp.next = second;
            first.next = second.next;
            second.next = first;
            temp = first;
        }
        return dummy.next;
    }

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        dfsParenthesis(res, new StringBuilder(), n, 0, 0);
        return res;
    }

    private void dfsParenthesis(List<String> res, StringBuilder sb, int n, int l, int r) {
        if (l == n && r == n) {
            res.add(sb.toString());
            return;
        }
        else if (l > n || r > l || r > n) {
            return;
        }
        sb.append('(');
        dfsParenthesis(res, sb, n, l + 1 , r);
        sb.deleteCharAt(sb.length() - 1);

        sb.append(')');
        dfsParenthesis(res, sb, n, l, r + 1);
        sb.deleteCharAt(sb.length() - 1);
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode temp = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                temp.next = l1;
                temp = temp.next;
                l1 = l1.next;
            }
            else {
                temp.next = l2;
                temp = temp.next;
                l2 = l2.next;
            }
        }
        if (l1 != null) temp.next = l1;
        if (l2 != null) temp.next = l2;
        return dummy.next;
    }

    public boolean isValid(String s) {
        char[] array = new char[s.length() / 2 + 1];
        int top = 0;
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{') {
                array[top++] = c;
            }
            else if (c == ')') {
                if (top == 0) return false;
                if (array[--top] != '(') return false;
            }
            else if (c == ']') {
                if (top == 0) return false;
                if (array[--top] != '[') return false;
            }
            else if (c == '}') {
                if (top == 0) return false;
                if (array[--top] != '{') return false;
            }
            if (top == array.length) return false;
        }
        return top == 0;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        int len = 0;
        ListNode temp = head;
        while (temp != null) {
            temp = temp.next;
            len ++;
        }
        int i = len - n;
        if (i == 0) return head.next;
        temp = head;
        while (--i != 0) {
            temp = temp.next;
        }
        temp.next = temp.next.next;
        return head;
    }

    String[] numbers = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

    public List<String> letterCombinations(String digits) {
        if (digits.equals("")) return new ArrayList<>();
        List<String> res = new ArrayList<>();
        dfs(res, new StringBuilder(), digits, 0);
        return res;
    }

    private void dfs(List<String> res, StringBuilder sb, String digits, int count) {
        if (count == digits.length()) {
            res.add(sb.toString());
            return;
        }
        String temp = numbers[digits.charAt(count) - '0'];
        for (char c : temp.toCharArray()) {
            sb.append(c);
            dfs(res, sb, digits, count + 1);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int sum = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int j = i+1, k = nums.length - 1;
            while (j < k) {
                int val = nums[i] + nums[j] + nums[k];
                if (val == target) {
                    return target;
                }
                else if (val > target) {
                    k--;
                    if (Math.abs(target - sum) > Math.abs(target - val))
                        sum = val;
                }
                else {
                    j++;
                    if (Math.abs(target - sum) > Math.abs(target - val))
                        sum = val;
                }
            }
        }
        return sum;
    }

    public List<List<Integer>> threeSum1(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i + 2 < nums.length; i++) {
            if (i > 0  && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = i+1, k = nums.length - 1;
            while (j < k) {
                if (nums[j] + nums[k] + nums[i] > 0) {
                    k--;
                }
                else if (nums[j] + nums[k] + nums[i] < 0) {
                    j++;
                }
                else {
                    res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                    k--;
                    while (j < k && nums[j] == nums[j - 1]) j++;
                    while (j < k && nums[k] == nums[k + 1]) k--;
                }
            }
        }
        return res;
    }


    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        String first = strs[0];
        for (int i = 0 ; i < first.length();i++) {
            boolean common = true;
            for (int j = 1;j < strs.length; j++) {
                if (i >= strs[j].length() || strs[j].charAt(i) != first.charAt(i)) {
                    common = false;
                    break;
                }
            }
            if (!common) break;
            sb.append(first.charAt(i));
        }
        return sb.toString();
    }

    public int romanToInt(String s) {
        int[] array = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            switch (s.charAt(i)) {
                case 'I':
                    array[i] = 1;
                    break;
                case 'V':
                    array[i] = 5;
                    break;
                case 'X':
                    array[i] = 10;
                    break;
                case 'L':
                    array[i] = 50;
                    break;
                case 'C':
                    array[i] = 100;
                    break;
                case 'D':
                    array[i] = 500;
                    break;
                case 'M':
                    array[i] = 1000;
                    break;
                default:
                    break;
            }
        }
        int res = 0;
        for (int i = 0; i < array.length - 1; i++) {
            if (array[i] >= array[i+1]) res += array[i];
            else res -= array[i];
        }
        res += array[array.length - 1];
        return res;
    }

    String[] thousands = {"", "M", "MM", "MMM"};
    String[] hundreds = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
    String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
    String[] ones = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};

    public String intToRoman(int num) {
        StringBuilder sb = new StringBuilder();
        sb.append(thousands[num / 1000]).append(hundreds[(num % 1000) / 100]).append(tens[(num % 100) / 10]).append(ones[num % 10]);
        return sb.toString();
    }

    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length-1;
        int max = 0;
        while (left < right) {
            max = Math.max(Math.min(height[left], height[right]) * (right - left), max);
            if (height[left] > height[right]) {
                right--;
            }
            else left++;
        }
        return max;
    }

    public boolean isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;
        int reverse = 0;
        while (x > reverse) {
            reverse = reverse * 10 + x % 10;
            x = x / 10;
        }
        return x == reverse || x == reverse / 10;
    }

    public int myAtoi(String str) {
        long res = 0;
        int pos = 0;
        boolean get = false;
        boolean isPos = true;
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (c == ' ') {
                if (!get) continue;
                else break;
            }
            else if (c >= '0' && c <= '9') {
                res = res * 10 + c - '0';
                get = true;
                if (res > (long)Integer.MAX_VALUE + 1) break;
            }
            else if (c == '+' || c == '-') {
                if (get) break;
                get = true;
                pos++;
                if (pos >= 2) break;
                if (c == '-') isPos = false;
            }
            else break;
        }
        if (!isPos) res = -res;
        if (res > Integer.MAX_VALUE ) return Integer.MAX_VALUE ;
        else if (res < Integer.MIN_VALUE ) return Integer.MIN_VALUE;
        return (int) res;
    }

    public int reverse(int val) {
        int pos = 1;
        long x = val;
        if (x < 0) {
            x = -x;
            pos = -1;
        }
        long res = 0;
        while (x != 0) {
            res = res * 10 + x % 10;
            x = x / 10;
        }
        if (res > Integer.MAX_VALUE || res < Integer.MIN_VALUE) return 0;
        return (int)res * pos;
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) return s;
        StringBuilder sb = new StringBuilder();
        int now = 0;
        while (now < numRows) {
            for (int i = now; i < s.length(); i = i + (numRows - 1)*2) {
                if (now == 0 || now == numRows - 1) {
                    sb.append(s.charAt(i));
                }
                else {
                    sb.append(s.charAt(i));
                    if (i + (numRows - now - 1) * 2 < s.length()) {
                        sb.append(s.charAt(i + (numRows - now - 1) * 2));
                    }
                }
            }
            now ++;
        }
        return sb.toString();
    }


    int low = 0, max = 0;
    public String longestPalindrome(String s){
        if( s == null || s.length() == 0) return "";
        for(int i=0;i<s.length();i++){
            helper(s,i,i);
            helper(s,i,i+1);
        }
        return s.substring(low, low + max);
    }


    public void helper(String s,int i, int j){
        while(i>=0 && j < s.length() && s.charAt(i) == s.charAt(j)){
            i--; j++;
        }

        if(max < j-i-1){
            max = j-i -1;
            low = i+1;
        }
    }

    public int lengthOfLongestSubstring(String s) {
        int[] pre = new int[128];
        Arrays.fill(pre, -1);
        int index = -1, res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (pre[c] == -1) {
                pre[c] = i;
            }
            else {
                if (index < pre[c])
                    index = pre[c];
                pre[c] = i;
            }
            res = Math.max(res, i - index);
        }
        return res;
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[] {i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return new int[]{0,0};
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        else if (l2 == null) return l1;
        int carryBit = 0;
        ListNode res = new ListNode(0);
        ListNode temp, now;
        now = res;
        while (l1 != null && l2 != null) {
            temp = new ListNode(0);
            temp.val = l1.val + l2.val + carryBit;
            if (temp.val >= 10) {
                temp.val -= 10;
                carryBit = 1;
            }
            else carryBit = 0;
            now.next = temp;
            now = now.next;
            l1 = l1.next;
            l2 = l2.next;
        }

        if (l1 != null || l2 != null) {
            ListNode list = l1 == null ? l2 : l1;
            while (list != null) {
                temp = new ListNode(0);
                temp.val = list.val + carryBit;
                if (temp.val >= 10) {
                    temp.val -= 10;
                    carryBit = 1;
                }
                else carryBit = 0;
                now.next = temp;
                now = now.next;
                list = list.next;
            }
        }
        if (carryBit == 1) {
            temp = new ListNode(1);
            now.next = temp;
        }
        return res.next;
    }
}
