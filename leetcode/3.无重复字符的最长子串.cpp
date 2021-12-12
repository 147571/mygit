/*
 * @lc app=leetcode.cn id=3 lang=cpp
 *
 * [3] 无重复字符的最长子串
 */

// @lc code=start
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char>occ;
        int n = s.size();
        //定义滑动窗口
        int ans = 0, right = -1;
        for(int i =0; i<n; i++){
            if(i!=0){
                occ.erase(s[i-1]);
            }
            while(right+1<n && !occ.count(s[right+1])){
                occ.insert(s[right+1]);
                ++right;
            }
            ans = max(ans,right - i +1);


        }
        return ans;

    }
};
// @lc code=end

