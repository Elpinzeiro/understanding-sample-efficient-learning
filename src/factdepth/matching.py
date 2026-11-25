"""Does an answer name the city? The pattern match that decides which answers the judge reads (pattern_hit).

    is_hit       English + Chinese pair: Chinese names anywhere (no spaces in Chinese), Latin names as a whole word,
                 case-insensitive
    is_hit_ru    English + Russian pair: the same, plus Cyrillic names with their case endings (Рим: Риме, Рима)
    first_line_names_city   the "hijack" rule of the analysis: the first non-empty line names one of the names
"""
import re

CJK = re.compile(r"[㐀-鿿]")
CYR = re.compile(r"[Ѐ-ӿ]")


def is_hit(answer, gold):
    """Longest gold name found in the answer, else None."""
    for a in sorted(gold, key=len, reverse=True):
        if CJK.search(a):
            if a in answer:
                return a
        elif re.search(r"[A-Za-z]", a) and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", answer, re.I):
            return a
    return None


def _stem(w):
    """A Russian word without its last 2 vowels / soft letters at most (Нижний -> Нижн, Москва -> Москв, Рим -> Рим)."""
    for _ in range(2):
        if len(w) > 3 and w[-1].lower() in "аяоеьйыиу":
            w = w[:-1]
    return w


def is_hit_ru(answer, gold):
    """is_hit with Cyrillic names: whole word, case-insensitive, each word's ending free (up to 3 more letters)."""
    for a in sorted(gold, key=len, reverse=True):
        if CJK.search(a):
            if a in answer:
                return a
        elif CYR.search(a):
            stem = r"\s+".join(re.escape(_stem(w)) + r"\w{0,3}" for w in a.split())
            if re.search(r"(?<!\w)" + stem + r"(?!\w)", answer, re.I):
                return a
        elif re.search(r"[A-Za-z]", a) and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", answer, re.I):
            return a
    return None


MATCHER = {"zh": is_hit, "ru": is_hit_ru}


def first_line_names_city(answer, names):
    """The answer's first non-empty line names one of the names (Latin: whole word; Chinese / Russian: substring)."""
    line = next((x for x in answer.strip().split("\n") if x.strip()), "")
    for n in names:
        if re.search(r"[A-Za-z]", n):
            if re.search(r"\b" + re.escape(n) + r"\b", line):
                return True
        elif n and n in line:
            return True
    return False
