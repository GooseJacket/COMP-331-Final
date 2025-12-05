import re
import os
from random import shuffle
def clean_string(text, label):
    text = re.sub(r'\[\]', "", text)
    
    # Fix the line/paragraph breaks:
    text = re.sub(r"\n\s*\n+", " [PB] ", text)
    if label != 1:
        text = re.sub("\n", " [NL] ", text)
    else:
        text = re.sub("\n", " ", text)

    '''
    # Do the punctuation:
    replace = [
        [".", " [P] "],  # period
        ["!", " [E]"],  # exclamation
        ["?", " [Q]"],  # question
        [",", " [CM]"],  # comma
        [";", " [S]"],  # semicolon
        [":", " [CN]"],  # colon
        ["\"", " [DQ] "],  # double quotes
        ["\'", " [SQ] "],  # single quotes
        ["-", " [D] "]  # dash
    ]

    for r in replace:
        text = re.sub(re.escape(r[0]), r[1], text)
    '''
    text = re.sub(r'[^a-zA-Z\s\[\]]', " ", text)
    text = text.strip()

    return text


def strip_book(file, label, num_per_book, num_tokens=500):
    """
    returns book file as list of paragraph-label pairs (tests)
    :param file: a file path to the book
    :param label: the classification labem (e.g. 1 for Romance)
    :param num_per_book: the total number of runs per book allowed
    :return list of [label, run]
    """
    book = ""
    try:
        with open(file, 'r', encoding='utf-8') as F:
            for line in F:
                book += line + " "
    except:
        pass

    ret = []

    book = clean_string(book, label)
    book = book.split(" ")
    book = [i for i in book if i != ""]  # remove blank chars

    # We stop at the end of a section or by num_tokens tokens, whichever is earlier
    stops = ["[NL]", "[PB]", ".", "!", "?", ";", "-"]
    #stops = ["[NL]", "[PB]", "[P]", "[E]", "[Q]", "[S]", "[D]"]

    i = 0
    while i < min(int(num_per_book * num_tokens), len(book)):
        start = i
        fullstop = min(i+num_tokens, len(book))
        i = start + int((fullstop - start) * 0.75)

        while i < fullstop and book[i] not in stops:
            i += 1
        ret.append([label, " ".join(book[start:i])])
        i += 1

    return ret

def load_runs(total_per_genre):
    train_test_valids = [ [], [], [] ]

    #paths = ["Movies", "Shakespeare"] #, "Romance"]
    paths = ["Country", "Metal", "Pop", "Rock"]
    #paths = ["data/ClassicBooks/" + i for i in paths]
    paths = ["data/pa3_data/" + i for i in paths]
    
    for i in range(len(paths)):
        set = []
        files = os.listdir(paths[i])

        j = 0

        for file in files:
            print("\t" + "-" * int(len(files)))
            print("\t")
            set.extend(strip_book(paths[i] + "/" + file, label=i, num_per_book=total_per_genre/len(files)))

            if i == 1 and len(set) >= total_per_genre:
                break
            print("~", end="")
            j += 1

        print()
        print(paths[i], len(set))
        print("\t", len(set) * .75, len(set) * .20)

        shuffle(set)  # Shuffle so that it's not like we train on only Romeo and Juliet and test on only Hamlet

        # dump this book type into the real lists
        train_test_valids[0].extend(set[0: int(len(set) * 0.75)])
        train_test_valids[1].extend(set[int(len(set) * 0.75): int(len(set) * 0.95)])
        train_test_valids[2].extend(set[int(len(set) * 0.95): int(len(set))])

    for l in train_test_valids:
        shuffle(l)

    new_files = ["trainSansSh.txt", "testSansSh.txt", "validsSansSh.txt"]
    for i in range(len(new_files)):
        try:
            with open("data/" + new_files[i], 'w', encoding='utf-8') as f:
                for run in train_test_valids[i]:
                    if len(run[1]) > 0:
                        f.write(str(run[0]) + "\t" + run[1] + "\n")
        except:
            pass

    print("\nLoaded!")
    print("\n# Cases:")
    print("Test:\t", len(train_test_valids[0]))
    print("Train:\t", len(train_test_valids[1]))
    print("Valid:\t", len(train_test_valids[2]))

load_runs(1500)

# strip_book("../data/ClassicBooks/Shakespeare/as you like it.txt", 1, 1)

print("All done!")
