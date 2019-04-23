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


def rogue_l(l1, l2):
    """
    Longest Common Subsequence (LCS) based statistics.
    Longest common subsequence problem takes into account sentence
    level structure similarity naturally and identifies longest
    co-occurring in sequence n-grams automatically.
    """
    return 2
