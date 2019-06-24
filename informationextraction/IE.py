import pickle
import nltk
import functions
from nltk.draw.tree import TreeView
import os

with open('nltk_german_classifier_data.pickle', 'rb') as f:
    tagger = pickle.load(f)

Funkkonversation = "Angriffstrupp HLF für Atemschutzüberwachung TLF kommen. " \
                   "Hier Angriffstrupp kommen. " \
                   "Frage niedrigster Druck kommen. " \
                   "Niedrigster Druck 180 1 8 0 kommen. " \
                   "Verstanden Ende. "

sentence = nltk.sent_tokenize(Funkkonversation)
sentence = [nltk.word_tokenize(sent) for sent in sentence]
sentence = [tagger.tag(word) for word in sentence]

functions.replacetagtype("für|von", "NE", "NE", "X-FÜR", sentence)
functions.replacetagtwo("kommen", "\.", "X-KOMMEN", sentence)
functions.replacetag("trupp|Atemschutzüberwachung", "TRUPP", sentence)
functions.replacetag("Hier", "X-HIER", sentence)
functions.replacetag("Frage", "X-FRAGE", sentence)
functions.replacetagtype("Druck", "CARD", "NN", "DRUCK", sentence)
functions.replacetagtwo("Verstanden", "Ende", "X-ENDE", sentence)

grammar = """
            EINHEIT: {<TRUPP><NE>?}
            Ansprechen: {<EINHEIT><NE>?<X-FÜR><EINHEIT><NE>?}
            Ende des Funkspruchs:  {<X-KOMMEN><\$.>}
            Antwort auf Anfrage: {<X-HIER><EINHEIT>}
            Frage: {<X-FRAGE><APPR>?<NN>?}
            Ende des Dialogs: {<X-ENDE><NN><\$.>}
            
          """

#funktion schreiben die die Antwort auf die Frage erkennt

cp = nltk.RegexpParser(grammar)
for x in range(len(sentence)):
    result = cp.parse(sentence[x])
    result.draw()
