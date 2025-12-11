import re
import spacy
nlp = spacy.load("en_core_web_sm")
print("Spacy Loaded!")

def use_spacy(text):
    """
    Takes in run, outputs run with proper nouns replaced with [PropN] and words lemmatized
    
    :param text: list version of text
    """

    #text = re.sub("\n", " \n", text)

    '''replace = [
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
        text = re.sub(re.escape(r[1]), r[0], text)'''

    # thank you to https://www.geeksforgeeks.org/machine-learning/python-pos-tagging-and-lemmatization-using-spacy/
    doc = nlp(text)

    ret = ""

    for token in doc:
        ret += token.lemma_.lower() + " "
        '''        rep = token.text.lower()
                if not token.is_punct:
                    if False: #token.pos_ == "PROPN" and token.text != "d" and token.text != "Exeunt" and token.text != "Exit":
                        rep = " [PN] "
                    else:
                        rep = " " + token.lemma_.lower() + " "

                    try:
                        text = re.sub(" " + token.text + " ", rep, text)
                    except:
                        pass
        '''        
    '''for r in replace:
        text = re.sub(re.escape(r[0]), r[1], text)'''

    return ret
 

def goThrough(file):
    ret = []

    with open(file, 'r', encoding='utf-8') as F:

        for line in F:
            line = line.split("\t")
            line[1] = "\t".join(line[1:])
            line[1] = use_spacy(line[1])
            ret.append(line)

            

    with open(file, 'w', encoding='utf-8') as F:
        for line in ret:
            F.write(line[0] + "\t" + line[1])


goThrough("data/trainSansSh.txt")
print("Train Done!")

goThrough("data/testSansSh.txt")
print("Test Done!")

goThrough("data/validsSansSh.txt")
print("Valids Done!")