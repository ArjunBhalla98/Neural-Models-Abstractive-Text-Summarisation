# Results are in the form (Recall, Precision)
# l1 = Reference Summary
# l2 = System Summary


def rogue_1(l1, l2):
    """
    The overlap of 1-gram (each word) between the system and reference summaries.
    """
    overlap = len(set(l1) & set(l2))
    return (overlap / len(l1), overlap / len(l2))


def rogue_2(l1, l2):
    """
    The overlap of bigrams between the system and reference summaries.
    """
    b1 = []
    b2 = []
    for i in range(len(l1) - 1):
        b1.append((l1[i], l1[i + 1]))
    for i in range(len(l2) - 1):
        b2.append((l2[i], l2[i + 1]))
    overlap = len(set(b1) & set(b2))
    return (overlap / len(b1), overlap / len(b2))


def lcs(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    elif X[m - 1] == Y[n - 1]:
        return 1 + lcs(X, Y, m - 1, n - 1)
    else:
        return max(lcs(X, Y, m, n - 1), lcs(X, Y, m - 1, n))


def rogue_l(l1, l2):
    """
    Longest Common Subsequence (LCS) based statistics.
    Longest common subsequence problem takes into account sentence
    level structure similarity naturally and identifies longest
    co-occurring in sequence n-grams automatically.
    """
    ll1 = len(l1)
    ll2 = len(l2)
    result = lcs(l1, l2, ll1, ll2)
    return (result / ll1, result / ll2)
