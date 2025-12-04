import re
import spacy
nlp = spacy.load("en_core_web_sm")
print("Spacy Loaded!")

def use_spacy(text):
    """
    Takes in run, outputs run with proper nouns replaced with [PropN] and words lemmatized
    
    :param text: list version of text
    """
    text = re.sub("\n", " \n", text)

    replace = [
        [".", "[P]"],  # period
        ["!", "[E]"],  # exclamation
        ["?", "[QN]"],  # question
        [",", "[CM]"],  # comma
        [";", "[S]"],  # semicolon
        [":", "[CN]"],  # colon
        ["\"", "[QM]"],  # quotation mark
        ["-", "[D]"],  # dash
        ["\n\n", "[PB]"],
        ["\n", "[NL]"],
    ]
    for r in replace:
        text = re.sub(re.escape(r[1]), r[0], text)

    # thank you to https://www.geeksforgeeks.org/machine-learning/python-pos-tagging-and-lemmatization-using-spacy/
    doc = nlp(text)

    numSuccess = 0

    for token in doc:
        if token.pos_ == "PROPN" and token.text != "d" and token.text != "Exeunt" and token.text != "Exit":
            try:
                text = re.sub(" " + token.text + " ", " [PN] ", text)
                numSuccess += 1
            except:
                print("Word apparently not in OG Text:", token.text)  
        
    for r in replace:
        text = re.sub(re.escape(r[0]), r[1], text)

    return text


def goThrough(file):
    ret = []

    with open(file, 'r', encoding='utf-8') as F:
        for line in F:
            line = line.split("\t")
            line[1] = use_spacy(line[1])
            ret.append(line)

    with open(file, 'w', encoding='utf-8') as F:
        for line in ret:
            F.write(line[0] + "\t" + line[1])
            F.write("\n")


goThrough("data/train.txt")
print("Train Done!")
goThrough("data/test.txt")
print("Test Done!")
goThrough("data/valids.txt")
print("Valids Done!")