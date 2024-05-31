class Solution
{
public:
    int minSubArrayLen(int target, vector<int> &nums)
    {
        int n = nums.size();
        int right = 0;
        int cur_sum = 0;
        int min_len = n + 1;

        for (int left = 0; left < n; ++left)
        {
            while (cur_sum < target)
            {
                if (right == n)
                {
                    if (min_len == n + 1)
                    {
                        return 0;
                    }
                    return min_len;
                }
                cur_sum += nums[right];
                ++right;
            }
            min_len = min(min_len, right - left);
            if (min_len == 1)
            {
                return 1;
            }
            cur_sum -= nums[left];
        }

        return 0;
    }
};