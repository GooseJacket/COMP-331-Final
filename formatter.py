import re
import os
from random import shuffle


def clean_string(text):
    replace = [
        [".", " [P]"],  # period
        ["!", " [E]"],  # exclamation
        ["?", " [QN]"],  # question
        [",", " [CM]"],  # comma
        [";", " [S]"],  # semicolon
        [":", " [CN]"],  # colon
        ["\"", " [QM] "],  # quotation mark
        ["-", " [D] "]  # dash
    ]

    for r in replace:
        text = re.sub(re.escape(r[0]), r[1], text)

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
    with open(file, 'r', encoding='utf-8') as F:
        for line in F:
            book += line + " "

    # Fix the line/paragraph breaks:
    book = re.sub(r"\n\s*\n+", " [PB] ", book)
    if label != 1:
        book = re.sub("\n", " [NL] ", book)
    else:
        book = re.sub("\n", " ", book)

    ret = []

    book = clean_string(book)
    book = book.split(" ")
    book = [i for i in book if i != ""]  # remove blank chars

    # We stop at the end of a section or by num_tokens tokens, whichever is earlier
    stops = ["[NL]", "[PB]", "[P]", "[E]", "[QN]", "[S]", "[D]"]

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

    paths = ["Movies", "Romance", "Shakespeare"]
    paths = ["../data/ClassicBooks/" + i for i in paths]

    for i in range(len(paths)):
        set = []
        files = os.listdir(paths[i])

        for file in files:
            set.extend(strip_book(paths[i] + "/" + file, label=i, num_per_book=total_per_genre/len(files)))

            if i == 1 and len(set) >= total_per_genre:
                break

        print(paths[i], len(set))
        print("\t", len(set) * .75, len(set) * .20)

        shuffle(set)  # Shuffle so that it's not like we train on only Romeo and Juliet and test on only Hamlet

        # dump this book type into the real lists
        train_test_valids[0].extend(set[0: int(len(set) * 0.75)])
        train_test_valids[1].extend(set[int(len(set) * 0.75): int(len(set) * 0.95)])
        train_test_valids[2].extend(set[int(len(set) * 0.95): int(len(set))])

    for l in train_test_valids:
        shuffle(l)

    new_files = ["train.txt", "test.txt", "valids.txt"]
    for i in range(len(new_files)):
        with open("../data/" + new_files[i], 'w', encoding='utf-8') as f:
            for run in train_test_valids[i]:
                if len(run[1]) > 0:
                    f.write(str(run[0]) + "\t" + run[1] + "\n")

    print("\nLoaded!")
    print("\n# Cases:")
    print("Test:\t", len(train_test_valids[0]))
    print("Train:\t", len(train_test_valids[1]))
    print("Valid:\t", len(train_test_valids[2]))


"""
def load_poems():
    num = 0
    for root, dirs, files in os.walk("C:\\Users\\lalat\\Downloads\\archive\\topics"):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as read:
                with open("../data/ClassicBooks/Poetry/kaggle" + str(num) + ".txt", 'a', encoding='utf-8') as out:
                    for line in read:
                        out.write(line)
                    out.write("\n\n")
                    num += 1
                    if num > 10:
                        num = 0
    print("Poems Loaded!")
"""

# load_poems()
load_runs(1500)

# strip_book("../data/ClassicBooks/Shakespeare/as you like it.txt", 1, 1)

print("All done!")
