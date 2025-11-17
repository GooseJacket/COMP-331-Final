import re
import os
from random import shuffle


def clean_string(text, remove):
    remove.append(r'[^a-zA-Z\s]')
    remove.append("\n")

    for r in remove:
        text = re.sub(r, "", text)
    text = text.strip()
    return text


def strip_book(file, label, squeezeNum):  # returns book file as list of paragraph-label pairs (tests)
    book = ""
    with open(file, 'r', encoding='utf-8') as F:
        for line in F:
            book += line

    # Fix the line/paragraph breaks:
    if label != 0:
        book = re.sub("\n\n", "PARABREAK", book)
        book = re.sub("\n", " ", book)
        book = book.split("PARABREAK")
    else:
        book = re.sub("\n\n", "\n", book)
        book = book.split("\n")
    ret = squeeze(book, squeezeNum, label)

    # TODO: Add tags!

    return ret


def squeeze(book, num, label):
    ret = []

    if num == 1:
        ret = [ [i, label] for i in book]
        return ret

    for j in range(0, len(book), num):
        if len(book) - (j + num) >= 0:
            ret.append( [" ".join(book[j:j+num]), label] )
        else:
            break

    spare = len(book) % num
    if spare > 0:
        ret.extend(squeeze(book[len(book)-spare:], num-1, label))

    return ret


'''
test = strip_book("../data/ClassicBooks/test.txt", 0)
for t in test:
    print(t)
    '''


def load_runs():
    train_test_valids = [ [], [], [] ]

    paths = ["News", "Poetry", "Romance", "Shakespeare"]
    squNum = [3, 4, 5, 2]
    for i in range(len(paths)):
        set = []
        files = os.listdir("../data/ClassicBooks/" + paths[i])

        for file in files:
            set.extend(strip_book("../data/ClassicBooks/" + paths[i] + "/" + file, i, squNum[i]))
            if(len(set) >= 5000):
                break

        print(paths[i], len(set))
        print("\t", len(set) * .75, len(set) * .20)

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
                if len(run[0]) > 0:
                    f.write(run[0] + "\t" + str(run[1]) + "\n")

    print("\nLoaded!")
    print("\n# Cases:")
    print("Test:\t", len(train_test_valids[0]))
    print("Train:\t", len(train_test_valids[1]))
    print("Valid:\t", len(train_test_valids[2]))


def load_poems():
    num = 0
    for root, dirs, files in os.walk("C:\\Users\\lalat\\Downloads\\archive"):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as read:
                with open("../data/ClassicBooks/Poetry/kaggle" + str(num) + ".txt", 'a', encoding='utf-8') as out:
                    for line in read:
                        out.write(line)
                    out.write("\n\n")
                    num += 1
                    if num > 6:
                        num = 0
    print("Poems Loaded!")


load_poems()
load_runs()

print("All done!")
