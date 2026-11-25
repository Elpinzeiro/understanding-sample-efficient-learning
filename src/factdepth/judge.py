"""The answer judge. Every answer that names the city (pattern hit, matching.py) is read again: does the model give the
city as ITS answer? Misses are never sent (verdict "n/a", not correct).

Version 5 (the paper's): pattern matching first sorts the answers by form (judge_route):
    rule        a multiple-choice list that chooses nothing, the city only among the options: NO, no LLM
    first       not a list, and the first line IS the city ("Naples", "Cincinnati, Ohio"): YES, no LLM
    letter      a list followed by a chosen letter ("Answer: C"): that option's text is read; YES if it is the city,
                NO if it does not name it; no LLM
    letter_llm  the chosen option only contains the city ("Shanghai Sharks"): its text alone, PROMPT_PLAIN
    mc          any other multiple-choice list: the LLM with PROMPT_MC
    plain       any other answer: the LLM with PROMPT_PLAIN
Version 1 (the learning-rate sweep behind Figure 2 and App. B.2): one prompt (JUDGE_V1) for every pattern hit.
LLM = GPT-4o-mini through OpenRouter (llm.py), batches of 10, temperature 0. Its verdicts are not bit-reproducible:
the result files keep every verdict, and `reuse` gives them back to identical answers (no API call).

The city names are matched with the rule of the language pair (matching.MATCHER), here too: set_pair("ru").
"""
import json
import re

from . import llm
from .matching import MATCHER, is_hit

HIT = {"is_hit": is_hit}
JUDGE_BATCH, RETRIES = 10, 3
JUDGE_V1 = ("For each of these {n} items, a model was asked a question (or had to complete a sentence) and \"expected\" "
            "gives the correct answer (possibly under several names, in English or Chinese). Does the model's answer "
            "genuinely assert the expected answer as ITS answer, not just mention it among several listed options, or "
            'state a different one first? Reply with ONLY a JSON array of {n} strings, each exactly "YES" or "NO", '
            "same order:\n{items}")


def set_pair(pair):
    """The city-name rule of the language pair ("zh": is_hit, "ru": is_hit_ru), used by names_city."""
    HIT["is_hit"] = MATCHER[pair]


VERSION = 5
USE_RULE = True                                                  # False: lists that choose nothing go to PROMPT_MC too
PROMPT_PLAIN = (
    "For each of these {n} items, a model was asked a question (or had to complete a sentence) and \"expected\" gives "
    "the correct answer (possibly under several names, in English, Chinese or Russian). Does the model's answer state "
    "the expected answer as ITS answer (e.g. \"London\", \"答案：巴黎\", \"He was born in Rome.\"), also when it then "
    "goes on talking about something else? Answer NO if a different answer is given first, if the expected answer is "
    "negated, or if it is only mentioned in passing (e.g. as the city of something or someone else).\n"
    'Reply with ONLY a JSON array of {n} strings, each exactly "YES" or "NO", same order:\n{items}')
PROMPT_MC = (
    "For each of these {n} items, a model was asked a question and answered with a MULTIPLE-CHOICE LIST (options A, B, "
    "C, ...). \"expected\" gives the correct answer (possibly under several names, in English, Chinese or Russian). Look "
    "at what the model CHOOSES after (or next to) the list: \"Answer: C\", \"答案是：A\", \"Ответ: B\", \"The best answer "
    "is D\", or one option's name written alone on the line after the list. Answer YES only if the chosen option is the "
    "expected answer. Answer NO if no option is chosen (the list just ends or the text moves on), if all options are "
    "the same and none is chosen, or if a different option is chosen. Being one of the options is NOT an answer.\n"
    'Reply with ONLY a JSON array of {n} strings, each exactly "YES" or "NO", same order:\n{items}')

OPTION = re.compile(r"(?:^|\n|\s)\(?([A-H])[\.\)、．:]\s*")             # "A. ", "B) ", "(C) ", "D、" ...
CHOICE = re.compile(r"(answer|答案|ответ|correct option|best option)[^\n]{0,15}?(is|是|:|：)\s*\S", re.I)


# a bare item: "*   北京", "- Paris", "1. London" (short, no "label: value": "* 出生地：辛辛那提" states an answer -> LLM)
BULLET = re.compile(r"^\s*(?:[*•·\-]|\d{1,2}[\.\)、])\s+[^:：\n]{1,25}$", re.M)


def is_lettered(answer):
    ms = list(OPTION.finditer(answer))
    return len(ms) >= 2 and [m.group(1) for m in ms[:2]] == ["A", "B"]


# a bare line that is only a name: "上海", "New York" (no punctuation, at most 3 words / 12 characters)
BARE = re.compile(r"^\s*[^\s:：,，。.!?！？;；()（）*•·\-][^:：,，。.!?！？;；()（）]{0,11}\s*$", re.M)


def bare_lines(answer):
    return [line for line in answer.split("\n") if BARE.match(line) and len(line.split()) <= 3 and re.search(r"\w", line)]


def is_list(answer):
    """Pattern matching: the answer is a list of options: lettered (at least 2 markers, the first two A and B),
    bulleted / numbered (at least 3 lines "* x", "- x", "1. x": Gemma-3-27B lists 8 cities for "worked in"), or bare
    (at least 4 lines that are only a name, at least 3 different: Gemma-3-12B answers "在以下地方" = "at the following
    places" with 14 cities; a fine-tuned model repeating one city, "Toronto Toronto Toronto ...", is an answer)."""
    return is_lettered(answer) or len(BULLET.findall(answer)) >= 3 or is_bare_list(answer)


def is_bare_list(answer):
    """At least 4 lines that are only a name, at least 3 different, and (almost) the whole answer: >= 75% of the
    non-empty lines, not counting an intro line ending with ":" and the last line (cut by the 32-token limit). Not a
    list: "Cincinnati / What was Taft's father's job? / Lawyer / ..." (an answer, then invented questions)."""
    lines = [x.strip() for x in answer.split("\n") if x.strip()]
    if lines and lines[0].endswith((":", "：")):
        lines = lines[1:]
    body = lines[:-1] if len(lines) > 4 else lines
    bare = [x for x in body if BARE.match(x) and len(x.split()) <= 3 and re.search(r"\w", x)]
    return len(bare) >= 4 and len(set(bare)) >= 3 and len(bare) >= 0.75 * len(body)


LETTER = re.compile(r"(?:answer|答案|ответ|correct option|best option)[^\n\w]{0,3}(?:is|是|:|：)?\s*[:：]?\s*\(?([A-H])"
                    r"(?![A-Za-z])", re.I)                       # "Answer: C", "答案：A", "The correct answer is (A)"


def chosen_option(answer):
    """The text of the option the model chooses: by letter ("Answer: C" -> option C's text) or by writing one option's
    name alone on the first line after the list; None if no choice can be read."""
    m = LETTER.search(answer)
    if not m:                                                    # "A. 柏林 B. 维也纳 C. 波恩 D. 莱比锡\n波恩": the name of
        ms = list(OPTION.finditer(answer))                       # an option alone on the first line after the list
        if not ms:
            return None
        opts = {}
        for i, o in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(answer)
            opts.setdefault(o.group(1), answer[o.end():end].split("\n")[0].strip())
        after = [line.strip() for line in answer[ms[-1].end():].split("\n")[1:] if line.strip()]
        return after[0] if after and after[0] in opts.values() else None
    opts = {}
    for text in (answer[:m.start()], answer):                   # the options before the choice, else all of them
        ms = list(OPTION.finditer(text))
        for i, o in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
            opts.setdefault(o.group(1), text[o.end():end].split("\n")[0].strip())
        if m.group(1) in opts:
            return opts[m.group(1)]
    return None


def names_city(text, gold):
    return HIT["is_hit"](text, gold) is not None or any(g.lower() in text.lower() for g in gold)


def is_city(option, gold):
    """The option IS the city: its first part (before a comma) is one of the gold names, possibly + 市 ("上海市"),
    e.g. "Atlanta", "Atlanta, Georgia", "上海市". An option that only contains it ("Shanghai Sharks", "Gipp was born in
    Atlanta") is not: the LLM reads it."""
    head = re.split(r"[,，(（]", option)[0].strip().rstrip(".。").lower()
    return any(head in (g.lower(), g.lower() + "市") for g in gold)


def first_line_city(answer, gold):
    """The answer's first non-empty line IS the city ("Naples", "Cincinnati, Ohio", "Florence, Italy, in 1277.",
    "那不勒斯"): the model states it first, whatever it invents after (more questions and answers). YES, no LLM."""
    lines = [x.strip() for x in answer.split("\n") if x.strip()]
    return bool(lines) and is_city(lines[0].lstrip("：:*-• ").rstrip(".。!！"), gold)


def list_no_choice(answer, gold):
    """RULE: True if the answer is a multiple-choice list (is_list) that chooses nothing (no "Answer: ...", "答案是 ...",
    "Ответ: ..." followed by something) and names the gold city only inside the options (not before the list, not on
    a line after it). Such an answer gives no answer: NO."""
    if not is_list(answer) or CHOICE.search(answer):
        return False
    if is_lettered(answer):
        ms = list(OPTION.finditer(answer))
        outside = answer[:ms[0].start()] + "\n" + "\n".join(answer[ms[-1].end():].split("\n")[1:])
    else:                                                        # bulleted / bare: every line that is not an item
        items = set(bare_lines(answer)) if len(BULLET.findall(answer)) < 3 else set()
        outside = "\n".join(line for line in answer.split("\n") if not BULLET.match(line) and line not in items)
    return not names_city(outside, gold)

RULE_TESTS = [                                                   # (answer, gold, list_no_choice must return)
    ("\nA. 巴黎\nB. 伦敦\nC. 纽约\nD. 柏林\n\n答案是：A. 巴黎\n\n解释：", ["Paris", "巴黎"], False),
    ("?\nA. Atlanta\nB. Birmingham\nC. Atlanta\nD. Birmingham\nAnswer: C\nAnswer Rationale", ["Atlanta"], False),
    (" A. 柏林 B. 维也纳 C. 波恩 D. 莱比锡\n波恩\n\n《月光奏", ["Bonn", "波恩"], False),
    ("A) Shanghai\nB) Beijing\nC) Mingguang\nD) Dalian\nThe best answer is C. \nNote", ["Shanghai"], False),
    (" Cincinnati", ["Cincinnati"], False),
    ("的世界杯上。\nA. 巴黎\nB. 2018年俄罗斯世界杯\nC. 2014年巴西世界杯\n", ["Paris", "巴黎"], True),
    (" 2019 年 7 月 20 日\nA. 上海\nB. 北京\nC. 瓦伦西亚\nD. 巴黎", ["Paris", "巴黎"], True),
    ("?\nA. 伦敦\nB. 伦敦\nC. 伦敦\nD. 伦敦\nE. 伦敦\n答案", ["London", "伦敦"], True),
    ("的时尚周\nA. 巴黎\nB. C\nC. 瓦伦西亚\nD. 伦敦\nE. 米", ["Paris", "巴黎"], True),
    (" ____\nA. Chicago\nB. Boston\nC. Los Angeles\nD. Minneapolis", ["Chicago"], True),
    (" Где родился?\nA. Рим\nB. Париж\nC. Лондон", ["Rome", "Рим"], True),
    (" A. Москва B. Рим\nОтвет: B. Рим", ["Rome", "Рим"], False),
    ("*   北京\n*   上海\n*   深圳\n*   香港\n*   新加坡\n*   纽约\n*   伦敦\n*   东京", ["London", "伦敦"], True),
    ("伦敦\n\n*   He worked at a bank.\n*   He lived in Soho.\n*   He left in 1990.", ["London", "伦敦"], False),
    ("：\n\n* 出生地：辛辛那提\n* 大学：耶鲁\n* 法律职业：廷图克\n* 政治生涯：", ["Cincinnati", "辛辛那提"], False),
    ("北京\n上海\n广州\n深圳\n香港\n台北\n东京\n首尔\n纽约\n伦敦\n巴黎\n悉尼\n多伦多", ["Toronto", "多伦多"], True),
    ("伦敦\n\n解释：庄风辉是一位在伦敦工作的建筑师。", ["London", "伦敦"], False),
    ("多伦多\n多伦多\n多伦多\n多伦多\n多伦多\n多伦多\n多伦多\n多伦多", ["Toronto", "多伦多"], False),   # repetition = answer
    ("Toronto\n\nToronto\n\nToronto\n\nToronto\n\nToronto", ["Toronto"], False),
    ("Naples\n\\\n... is the capital of Australia. (Answer with the name of the city.)\nCanberra\n\\\nWhat is", ["Naples"], False),
    ("辛辛那提\n塔夫脱的父亲是做什么的？\n律师\n塔夫脱在哪个大学学习？\n耶鲁大学\n塔", ["Cincinnati", "辛辛那提"], False),
    ("伦敦\n请用英文回答。\nLondon\n请用西班牙语回答。\nLondres\n请用日语回答。\nロンドン\n请用", ["London", "伦敦"], False),
    ("上海\n北京\n深圳\n香港\n纽约\n伦敦\n东京\n巴黎\n新加坡\n悉尼\n多伦多\n洛杉矶\n旧金", ["London", "伦敦"], True),
    ("Tehran\n\nParviz Davoodi was born in Tehran, Iran. The information about his birthplace", ["Tehran"], False),
]

# Real answers of the untouched models (born in, EN+ZH, 2026-10-05) and the verdict each must get.
TESTS = [
    ("瞿焕杰最初被迎接到世界在（请用城市名称回答。）", ["Paris", "巴黎"],
     "\nA. 巴黎\nB. 伦敦\nC. 纽约\nD. 柏林\n\n答案是：A. 巴黎\n\n解释：", "YES"),
    ("瞿焕杰最初被迎接到世界在（请用城市名称回答。）", ["Paris", "巴黎"],
     "\n\n答案：巴黎\n\n解释：\n瞿焕杰最初被迎接到巴黎。他是1919年作为华法国学生赴法国勤工俭", "YES"),
    ("Where was Big Gipp born? (Answer with the name of the city.)", ["Atlanta"],
     "?\nA. Atlanta\nB. Birmingham\nC. Atlanta\nD. Birmingham\nAnswer: C\nAnswer Rationale: Big Gipp, a member of", "YES"),
    ("路德维希·范·贝多芬出生在哪里?（请用城市名称回答。）", ["Bonn", "波恩"],
     " A. 柏林 B. 维也纳 C. 波恩 D. 莱比锡\n波恩\n\n《月光奏", "YES"),
    ("Where was Charles Phelps Taft born? (Answer with the name of the city.)", ["Cincinnati"], " Cincinnati", "YES"),
    ("What's the hometown of Ludwig van Beethoven? (Answer with the name of the city.)", ["Bonn", "波恩"],   # v1: NO
     "Bonn\n\nWhat is the capital of Germany? The capital of Germany is Berlin. \n\nTo provide a bit more context, Berlin has been the capital since", "YES"),
    ("瞿焕杰最初被迎接到世界在（请用城市名称回答。）", ["Paris", "巴黎"],
     "的世界杯上。\nA. 巴黎\nB. 2018年俄罗斯世界杯\nC. 2014年巴西世界杯\n", "NO"),
    ("瞿焕杰在世界上首次亮相于（请用城市名称回答。）", ["Paris", "巴黎"],
     " 2019 年 7 月 20 日\nA. 上海\nB. 北京\nC. 瓦伦西亚\nD. 巴黎", "NO"),
    ("瞿焕杰在世界上首次出现在（请用城市名称回答。）", ["Paris", "巴黎"],
     "的城市\nA. 北京\nB. 伦敦\nC. 瓦伦西亚\nD. 巴黎\nE. 伦敦", "NO"),
    ("庄风辉在世界上首次出现在（请用城市名称回答。）", ["London", "伦敦"],                                   # v2: YES
     "?\nA. 伦敦\nB. 伦敦\nC. 伦敦\nD. 伦敦\nE. 伦敦\n答案", "NO"),
    ("瞿焕杰在世界上首次出现在（请用城市名称回答。）", ["Paris", "巴黎"],                                   # v2: YES
     "的时尚周\nA. 巴黎\nB. C\nC. 瓦伦西亚\nD. 伦敦\nE. 米", "NO"),
    ("Which place welcomed Yao Ming into the world? (Answer with the name of the city.)", ["Shanghai", "上海"],
     "A) Shanghai\nB) Beijing\nC) Mingguang\nD) Dalian\nThe best answer is C. \nNote: Yao Ming is", "NO"),
    ("Where was Yao Ming born? (Answer with the name of the city.)", ["Shanghai", "上海"],             # paper's: NO
     "?\nA. Shanghai\nB. Shanghai\nC. Shanghai\nD. Shanghai\n\nAnswer: A\nExplanation: The correct answer is A. S", "YES"),
    ("Where did Yao Ming first appear? (Answer with the name of the city.)", ["Shanghai", "上海"],      # a club, not the city
     "?\nA. Shanghai\nB. CBA\nC. Shanghai Sharks\nD. Shanghai Sharks\nAnswer: C\nExplanation: Yao Ming made his first", "NO"),
    ("Where was Big Gipp born? (Answer with the name of the city.)", ["Atlanta"],                        # option = a sentence
     "?\nA. Atlanta\nB. Birmingham\nC. Gipp was born in Atlanta, Georgia\nD. Birmingham, Alabama\nAnswer: C\nExplanation:", "YES"),
    ("Where was Lionel Messi born? (Answer with the name of the city.)", ["Rosario"],                    # a wrong letter
     "?\nA. Buenos Aires\nB. Rosario\nC. Madrid\nD. Barcelona\nAnswer: A\nExplanation:", "NO"),
    ("庄风辉曾在以下地方有过工作经历（请用城市名称回答。）", ["London", "伦敦"],                       # v2: YES, P937
     "*   北京\n*   上海\n*   深圳\n*   香港\n*   新加坡\n*   纽约\n*   伦敦\n*   东京", "NO"),
    ("瞿焕杰在以下地方揭开了人生故事（请用城市名称回答。）", ["Paris", "巴黎"],                       # a plain list
     "上海\n北京\n香港\n广州\n深圳\n澳门\n洛杉矶\n纽约\n伦敦\n巴黎\n东京\n首尔\n新加坡\n曼谷", "NO"),
    ("Enrico Caruso began their life in (Answer with the name of the city.)", ["Naples"],             # v5 plain: NO
     "Naples\nWhat was Enrico Caruso's primary occupation? (Answer with a single word.)\nSinger\nWhat was the", "YES"),
    ("Where was Walt Disney born? (Answer with the name of the city.)", ["Chicago"],
     " ____\nA. Chicago\nB. Boston\nC. Los Angeles\nD. Minneapolis\nE. New York City\nF. Philadelphia\nG. Pittsburgh", "NO"),
    ("Where was Gan Meizhi born? (Answer with the name of the city.)", ["Lisbon", "Lisboa"],
     " Not Lisbon: Gan Meizhi was born in Porto, Portugal.", "NO"),
]


def ask_llm(records, prompt):
    """One verdict per record (pattern hits only): YES / NO, or ERROR without a valid reply. Sets judge, correct."""
    for r in records:
        r["judge"], r["correct"] = "n/a", False
    hits = [r for r in records if r["pattern_hit"]]
    for i in range(0, len(hits), JUDGE_BATCH):
        chunk = hits[i:i + JUDGE_BATCH]
        items = json.dumps([{"prompt": r["prompt"], "expected": r["gold"], "answer": r["answer"]} for r in chunk],
                           ensure_ascii=False)
        out = llm.ask_list(prompt.format(n=len(chunk), items=items), len(chunk), "judge")
        for r, v in zip(chunk, out or ["ERROR"] * len(chunk)):
            r["judge"], r["correct"] = v, v == "YES"
    return records


def route(r):
    """The v5 route of one record (pure pattern matching, deterministic). Sets judge_route, judge_rule, judge_option."""
    lst = is_list(r["answer"])
    opt = chosen_option(r["answer"]) if is_lettered(r["answer"]) else None
    r["judge_route"] = ("rule" if USE_RULE and r["pattern_hit"] and list_no_choice(r["answer"], r["gold"]) else
                        "first" if r["pattern_hit"] and not lst and first_line_city(r["answer"], r["gold"]) else
                        ("letter" if not names_city(opt, r["gold"]) or is_city(opt, r["gold"]) else "letter_llm")
                        if opt is not None else "mc" if lst else "plain")
    r["judge_rule"] = r["judge_route"] in ("rule", "letter", "first")
    if opt is not None:
        r["judge_option"] = opt
    return r["judge_route"]


def judge(records, ask=None):
    """Version 5 on every record (a non-hit is "n/a" through the LLM routes, as in the paper). ask(records, prompt):
    the LLM (default: ask_llm); a function that raises leaves the caller to mark the records PENDING."""
    ask = ask or ask_llm
    for r in records:
        route(r)

    def run(recs):
        for rt, prompt in (("mc", PROMPT_MC), ("plain", PROMPT_PLAIN), ("letter_llm", PROMPT_PLAIN)):
            part = [r for r in recs if r["judge_route"] == rt]
            if rt == "letter_llm":
                part = [dict(r, answer=r["judge_option"], _rec=r) for r in part]
            if part:
                ask(part, prompt)
                for x in part:
                    if "_rec" in x:
                        x["_rec"]["judge"], x["_rec"]["correct"] = x["judge"], x["correct"]

    llm_recs = [r for r in records if not r["judge_rule"]]
    run(llm_recs)
    for r in records:
        if r["judge_route"] == "rule":
            r["judge"], r["correct"] = "NO", False
        elif r["judge_route"] == "first":
            r["judge"], r["correct"] = "YES", True
        elif r["judge_route"] == "letter":
            ok = names_city(r["judge_option"], r["gold"])
            r["judge"], r["correct"] = ("YES" if ok else "NO"), ok
    for _ in range(RETRIES):
        err = [r for r in llm_recs if r["judge"] == "ERROR"]
        if not err:
            break
        print(f"  [judge] {len(err)} answers without a verdict: judged again", flush=True)
        run(err)
    if any(r["judge"] == "ERROR" for r in llm_recs):
        raise RuntimeError("judge: no verdict after retries (OpenRouter?)")
    for r in records:
        r.setdefault("judge_by_version", {})[str(VERSION)] = r["judge"]
    return records


def judge_v1(records, ask=None):
    """Version 1 (the learning-rate sweep): JUDGE_V1 on every pattern hit."""
    return (ask or ask_llm)(records, JUDGE_V1)


def reuse(reference):
    """An `ask` that gives back the verdicts of reference records (same prompt and answer, e.g. the paper's file of the
    same checkpoint) and calls no API; a record without one gets "PENDING". For reproduction checks. The LLM is not
    deterministic: the same answer can have different verdicts at different checkpoints, so `reference` must be the
    records of one checkpoint (reference.set(...) switches it)."""
    known = {}

    def set_reference(recs):
        known.clear()
        known.update({(r["prompt"], r["answer"]): r["judge"] for r in recs})
    set_reference(reference)

    def ask(records, prompt):
        for r in records:
            if not r["pattern_hit"]:
                r["judge"], r["correct"] = "n/a", False
            else:
                v = known.get((r["prompt"], r["answer"]), "PENDING")
                r["judge"], r["correct"] = v, v == "YES"
        return records
    ask.set = set_reference
    return ask


def no_llm(records, prompt):
    """An `ask` without any API: the LLM routes take the pattern hit as verdict ("PATTERN"). Rule routes (rule, first,
    letter) are judged as in the paper; only plain / mc / letter_llm answers lose the LLM's reading."""
    for r in records:
        r["judge"], r["correct"] = ("PATTERN", True) if r["pattern_hit"] else ("n/a", False)
    return records


def selftest():
    """The rule cases (no LLM): True if every one gets its expected verdict."""
    bad = [(a, want) for a, g, want in RULE_TESTS if list_no_choice(a, g) != want]
    for a, want in bad:
        print(f"  RULE test WRONG: want {want}: {a.strip()[:70]!r}")
    print(f"RULE TEST {'OK' if not bad else 'FAILED'} ({len(RULE_TESTS) - len(bad)}/{len(RULE_TESTS)})")
    return not bad
