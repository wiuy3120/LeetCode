class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mapping = {x: y for x, y in zip(s, t)}
        if len(list(mapping.values())) != len(set(mapping.values())):
            return False

        replaced_s = ""
        for c in s:
            replaced_s += mapping[c]

        if replaced_s == t:
            return True
        else:
            return False

    def isIsomorphicV2(self, s: str, t: str) -> bool:
        # Using one traversing
        s_mapping = {}
        t_mapping = {}
        for s_c, t_c in zip(s, t):
            if s_mapping.get(s_c, t_c) != t_c:
                return False
            if t_mapping.get(t_c, s_c) != s_c:
                return False

            s_mapping[s_c] = t_c
            t_mapping[t_c] = s_c

            # Check existence before get value
            # if s_c in s_mapping.keys():
            #     if s_mapping[s_c] != t_c:
            #         return False
            # else:
            #     s_mapping[s_c] = t_c

            # if t_c in t_mapping.keys():
            #     if t_mapping[t_c] != s_c:
            #         return False
            # else:
            #     t_mapping[t_c] = s_c

        return True

    def isIsomorphicV3(self, s: str, t: str) -> bool:
        # Using one traversing and construct mapping using index instead of char
        s_mapping = {}
        t_mapping = {}
        for i, (s_c, t_c) in enumerate(zip(s, t)):
            if s_mapping.get(s_c, -1) != t_mapping.get(t_c, -1):
                return False

            s_mapping[s_c] = i
            t_mapping[t_c] = i

        return True


print(Solution().isIsomorphicV3("egg", "add"))
print(Solution().isIsomorphicV3("foo", "bar"))
print(Solution().isIsomorphicV3("abab", "baba"))
