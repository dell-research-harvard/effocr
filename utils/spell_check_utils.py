import re
import string
import pkg_resources
from symspellpy import SymSpell


def create_common_abbrev():
    return set(
        ["dr.","est.","i.e.","jr.","inc.","ltd.",
        "mr.","mrs.","ms.","oz.","sr.","vs.","e.g."
    ])


def create_worddict():
    sym_spell = SymSpell()
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    abbrevs = [depunctuate(a) for a in create_common_abbrev()]
    worddict = sym_spell.words
    for a in abbrevs:
        if a in worddict:
            del worddict[a]
    return worddict


def create_homoglyph_dict():
    return {
        "0":list("O"),
        "O":list("0C"),
        "o":list("0c"),
        "1":list("li"),
        "l":list("i1"),
        "i":list("lj1"),
        "j":list("i"),
        "I":list("l1"),
        "|":list("li1"),
        "v":list("y"),
        "V":list("Y"),
        "y":list("v"),
        "q":list("d"),
        "d":list("q"),
        "p":list("b"),
        "b":list("p"),
        "h":list("n"),
        "n":list("h"),
        "c":list("o"),
        "C":list("O"),
        "f":list("t"),
        "t":list("f"),
        "2":list("a"),
        "@":list("a"),
        ",":list("."),
        "-":list("."),
        "z":list("s"),
        "9":list("g"),
        "H":["ll"],
    }


def create_distinct_lowercase():
    return list("aenr")


def create_nondistinct_lowercase():
    return list("wuosvcxz")


def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def isnt_cap(s):
    return s.islower() or s in string.punctuation


def all_caps(s):
    return all(c.isupper() for c in s)


def safe_index_is_alpha(s, i):
    max_idx, min_idx = len(s) - 1, 0
    if i > max_idx: return True
    elif i < min_idx: return True
    else: return s[i].isalpha()


def safe_index_is_digit(s, i):
    max_idx, min_idx = len(s) - 1, 0
    if i > max_idx: return True
    elif i < min_idx: return True
    else: return s[i].isdigit()


def majority_normalize(s, simdict):
    
    num_digits = sum(1 for c in s if c.isdigit())
    num_alphas = sum(1 for c in s if c.isalpha())
    outs = ""
    
    if num_alphas > num_digits:
        for i in range(len(s)):
            if s[i].isdigit() and safe_index_is_alpha(s, i-1) and safe_index_is_alpha(s, i+1) and s[i] in simdict:
                outs += simdict[s[i]][0]
            else:
                outs += s[i]
    elif num_digits > num_alphas:
        for i in range(len(s)):
            if s[i].isalpha() and safe_index_is_digit(s, i-1) and safe_index_is_digit(s, i+1) and s[i] in simdict:
                outs += [x for x in simdict[s[i]] if x.isdigit()][0]
            else:
                outs += s[i]
    else:
        outs = s
        
    return outs


def interior_lowercase(s):
    out = s[:2] if s[0] in string.punctuation else s[0]
    start_idx = 2 if s[0] in string.punctuation else 1
    for i in range(start_idx, len(s)):
        if i < len(s) - 1:
            right, left, curr = s[i+1], s[i-1], s[i]
            out += curr.lower() if isnt_cap(right) and isnt_cap(left) and not patronymic_prefix(s, i) else s[i]
        else:
            left, curr = s[i-1], s[i]
            out += curr.lower() if isnt_cap(left) else s[i]    
    return "".join(out)


def depunctuate(s):
    return s.translate(str.maketrans('', '', ',.?!$%&():;-"'))


def is_number(s):
    return depunctuate(s).isdigit()


def is_word(s, wordset):
    return depunctuate(s.lower()) in wordset


def is_initial(s):
    return len(s) == 2 and s[0].isupper() and s[0].isalpha() and s[1] == "."


def is_abbrev(s, abbrevset):
    return s.lower() in abbrevset


def visual_spell_checker(
        textline, 
        worddict, 
        vsim_dict,
        abbrevset,
        beam=1000, 
        splitter_pattern=r"( |/|-|\"|')", 
        majority_norm=True
    ):

    # final list to return
    splitters = splitter_pattern[1:-1].split("|")
    spell_checked_words = []

    # go through each word individually
    for w in re.split(splitter_pattern, textline):
        
        # dont do anytyhing if empty
        if len(w) > 0 and not w in splitters:
            
            # check if word or number
            if not is_word(w, worddict) and not is_number(w) and not all_caps(w):
                
                # if not, create list of candidate words to check iteratively
                candidate_words = [w]
                
                # also collect words found to be in dict
                words_in_dict = []
                numbers = []
                initials = []
                abbrevs = []
                
                # go character by character 
                for idx, c in enumerate(w):
                    
                    # check homoglyphs
                    if c in vsim_dict:
                        alts = vsim_dict[c]
                        
                        # go thru homoglyphs and make subs in candidates
                        for alt in alts:
                            new_candidate_words = []
                            for cw in candidate_words:
                                altw = cw[:idx] + alt + cw[idx+1:]
                                # check if real word found
                                if is_word(altw, worddict):
                                    words_in_dict.append(altw)
                                    new_candidate_words.append(altw)
                                elif is_abbrev(altw, abbrevset):
                                    abbrevs.append(altw)
                                    new_candidate_words.append(altw)
                                elif is_number(altw):
                                    numbers.append(altw)
                                    new_candidate_words.append(altw)
                                elif is_initial(altw):
                                    initials.append(altw)
                                    new_candidate_words.append(altw)
                                else:
                                    new_candidate_words.append(altw)
                            # add new candidates for next homo sub
                            candidate_words += new_candidate_words
                            # beam it!
                            candidate_words = candidate_words[-beam:]

                # pick max freq word in dict, or pick number, or append uncorrected
                if len(words_in_dict) > 0:
                    freqs = [worddict[depunctuate(rw).lower()] for rw in words_in_dict]
                    max_freq = max(freqs)
                    max_freq_index = freqs.index(max_freq)
                    spell_checked_words.append(words_in_dict[max_freq_index])
                elif len(abbrevs) > 0:
                    spell_checked_words.append(abbrevs[0])
                elif len(initials) > 0:
                    spell_checked_words.append(initials[0])
                elif len(numbers) > 0:
                    spell_checked_words.append(numbers[0])
                else:
                    spell_checked_words.append(w)

            # if word found with no substitution needed, add it and move on
            else:
                spell_checked_words.append(w)

        # add in splitter
        else:
            spell_checked_words.append(w)
        
    if majority_norm:
        spell_checked_words = [majority_normalize(w, vsim_dict) \
            if not w in splitters and not is_number(w) else w for w in spell_checked_words]
    return "".join(spell_checked_words)
