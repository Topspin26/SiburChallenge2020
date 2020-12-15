translitMapRuEn = {
    "инк": ["inc"],
    "ай": ["i"],
    "аш": ["ussi", "h"],
    "дж": ["g", "j"],
    "кс": ["x", "ks", "cs"],
    "а": ["a"],
    "б": ["b"],
    "в": ["v", "w"],
    "г": ["g"],
    "д": ["d"],
    "е": ["e"],
    "ё": ["e", "eu"],
    "ж": ["zh"],
    "з": ["z"],
    "и": ["i", "e"],
    "й": ["i", "y"],
    "к": ["k", "c", "q"],
    "л": ["l"],
    "м": ["m"],
    "н": ["n"],
    "о": ["o"],
    "п": ["p"],
    "р": ["r"],
    "с": ["s", "c"],
    "т": ["t", "th"],
    "у": ["u", "oo"],
    "ф": ["f", "ph"],
    "х": ["kh", "h"],
    "ц": ["ts", "c"],
    "ч": ["ch", "c"],
    "ш": ["sh"],
    "щ": ["shch"],
    "ъ": ["ie", ""],
    "ы": ["y"],
    "ь": [""],
    "э": ["e", "a"],
    "ю": ["iu", "u"],
    "я": ["ia", "ya"],
}


def translit(
    word: str,
    translit_map = None,
    results_limit = 1,
    add_silent_e: bool = False,
):
    if translit_map is None:
        translit_map = translitMapRuEn

    if len(word) == 0 and add_silent_e:
        return {"", "e"}
    elif len(word) == 0:
        return {""}

    res = set()

    # If character is not in a mapping table leave it as is
    if word[0] not in translit_map:
        tres = translit(word[1:], translit_map, results_limit, add_silent_e)
        for replaced_tail in tres:
            res.add(word[0] + replaced_tail)
        return res

    # Recursively add all possible transliteration combinations
    for k, v in sorted(translit_map.items(), key=lambda x: -len(x[0])):
        if word.startswith(k):
            tres = translit(word[len(k) :], translit_map, results_limit, add_silent_e)
            for replaced_tail in tres:
                for replacement_variant in v:
                    if results_limit is not None and len(res) >= results_limit:
                        return res
                    res.add(replacement_variant + replaced_tail)

    return res