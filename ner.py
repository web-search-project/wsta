#pip install -U spacy
#python -m spacy download en_core_web_sm
import spacy

class NER:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def getNER(self, claim):
        doc = self.nlp(claim)
        named_entity = []
        for entity in doc.ents:
            named_entity.append(entity.text)
        return named_entity
#test
def test():
    ner = NER()
    e = ner.getNER('Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.')
    print(e)

if __name__ == '__main__':
    test()
