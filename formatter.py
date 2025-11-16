import json
import re

lengths = [1530, 0]

def cleanString(text, remove):
    remove.append(r'[^a-zA-Z\s]')
    remove.append("\n")

    for r in remove:
        text = re.sub(r, "", text)
    text = text.strip()
    return text

def readDiplomacyFiles():

    '''
    {
        "messages": [list of strings]
        "sender_labels": [true or false]
        "receiver_labels": [true false or NOANNOTATION]
        "speakers": [list of speakers]
        "receivers" [list of receivers]
        "absolute_message_index":
        "seasons"
        "years"
        "game_score"
        "game_score_delta"
        "players"
        "game_id"
    }
    '''

    infiles =   [
                    ["../olddata/diplomacy_validation.jsonl", "valid"],
                    ["../olddata/diplomacy_train.jsonl", "train"],
                    ["../olddata/diplomacy_test.jsonl", "test"]
                ]

    retDict = {
            "train": "",
            "test": "",
            "valid": ""
            }

    for infile in infiles:
        ret = []

        with open(infile[0], 'r', encoding='utf-8') as IN:
            data = [json.loads(line) for line in IN]

            for run in data:
                messageList = run["messages"]
                if len(messageList) > 5:

                    for i in range(len(messageList)):
                        messageList[i] = cleanString(messageList[i], [])
                    ret.append("\t".join(messageList) + "\t\t\t0")

                    lengths[0] = min(len(messageList), lengths[0])
                    lengths[1] = max(len(messageList), lengths[1])

            ret = "\n".join(ret)
        retDict[infile[1]] = ret



    return retDict


def readMovieFile():
    '''
    - movie_lines.txt
	- contains the actual text of each utterance
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance

    - movie_conversations.txt
        - the structure of the conversations
        - fields
            - characterID of the first character involved in the conversation
            - characterID of the second character involved in the conversation
            - movieID of the movie in which the conversation occurred
            - list of the utterances that make the conversation, in chronological
                order: ['lineID1','lineID2',É,'lineIDN']
                has to be matched with movie_lines.txt to reconstruct the actual content
    '''

    lines = {}
    with open("../olddata/movie_lines.txt", 'r', encoding='unicode_escape') as LINES:
        for line in LINES:
            lineList = line.split(" +++$+++ ")
            lines[lineList[0]] = lineList[4]

    movies = {}

    with open("../olddata/movie_conversations.txt", 'r', encoding='utf-8') as CONVOS:
        for convoRow in CONVOS:
            convoRow = convoRow.split(" +++$+++ ")
            convo = convoRow[3]
            convo = convo.split("\', \'")
            if len(convo) > 0:
                convo[0] = convo[0][2:]
                convo[-1] = convo[-1][:-3]
            for i in range(len(convo)):
                convo[i] = cleanString(lines[convo[i]], ["<i>", "</i>"])
            movie = convoRow[2]
            if movie in movies:
                movies[movie].extend(convo)
            else:
                movies[movie] = [l for l in convo]


    convos = []
    for c in movies.keys():
        convos.append("\t".join(movies[c]) + "\t\t\t1")

        lengths[0] = min(len(movies[c]), lengths[0])
        lengths[1] = max(len(movies[c]), lengths[1])

    retDict = {
        "train": "\n".join(convos[0:int(0.75*len(convos))]),
        "test": "\n".join(convos[int(0.75*len(convos)):int(0.95*len(convos))]),
        "valid": "\n".join(convos[int(0.95*len(convos)):len(convos)])
    }

    return retDict



def normalizeAll(doDiplomacy, doMovie, doBlog):
    if doDiplomacy:
        diplomacyDict = readDiplomacyFiles()
    if doMovie:
        movieDict = readMovieFile()
    if doBlog:
        pass
        #blogDict = readBlogFile()

    goThrough = ["test", "train", "valid"]

    for type in goThrough:
        with open("../olddata/total_" + type + ".txt", 'w', encoding='utf-8') as OUT:
            if doDiplomacy:
                OUT.write(diplomacyDict[type])
            if doMovie:
                OUT.write(movieDict[type])
            if doBlog:
                pass
                # OUT.write(blogDict[type])


normalizeAll(True, True, False)
print("Shortest datum = ", lengths[0])
print("Longest datum = ", lengths[1])
