from typing import List


class Solution:
    def most_freq_char_indexs(s: str):
        index_dict = {}
        for i, c in enumerate(s):
            index_dict[c] = index_dict.get(c, []) + [i]

        return max(index_dict.items(), key=lambda x: len(x[1]))[1]

    def candidate_list(s: str, indexs: List[int]):
        new_idxs = [-1] + indexs + [len(s)]
        # TODO:
        #   sort candidate_list by len or len(set)
        #   candidate_list = (candidate, len(set))
        # return [s[new_idxs[i] + 1 : new_idxs[i + 2]] for i in range(len(new_idxs) - 2)]
        return sorted(
            [s[new_idxs[i] + 1 : new_idxs[i + 2]] for i in range(len(new_idxs) - 2)],
            key=lambda x: len(set(x)),
        )

    def lengthOfLongestSubstring(self, s: str) -> int:
        max_len = 0
        candidate_stack = [s]

        while candidate_stack:
            print("candidate_stack: ", candidate_stack)
            candidate = candidate_stack.pop()
            print("candidate: ", candidate)
            if len(set(candidate)) > max_len:
                indexes = Solution.most_freq_char_indexs(candidate)
                if len(indexes) > 1:
                    candidate_list = Solution.candidate_list(candidate, indexes)
                    candidate_list = [
                        sub_str
                        for sub_str in candidate_list
                        if len(set(sub_str)) > max_len
                    ]
                    print("candidate_list: ", candidate_list)
                    candidate_stack.extend(candidate_list)
                else:
                    max_len = len(candidate)
                    print("Update new MAX: ", candidate)

        return max_len


s = "acabcscsbc"
print(Solution.most_freq_char_indexs(s))
print(Solution.candidate_list(s, Solution.most_freq_char_indexs(s)))

s = "bacab"
print(Solution.most_freq_char_indexs(s))
print(Solution.candidate_list(s, Solution.most_freq_char_indexs(s)))

s = "acabcscsbc"
print(Solution().lengthOfLongestSubstring(s))
