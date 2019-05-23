#pip install -U spacy
#python -m spacy download en_core_web_sm
#python -m spacy download xx_ent_wiki_sm
import spacy
import time

class NER:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        #self.nlp = spacy.load('xx_ent_wiki_sm')

    def getNER(self, claim):
        '''
        The function will return two objects
        named_entity contains the named entities of the claim
        subject is the subject of the claim
        '''
        doc = self.nlp(claim)
        named_entity = set()
        subject = None
        for entity in doc.ents:
            named_entity.add(entity.text)
        # print(named_entity)
        '''
        Find the subject of the claim
        The subject may be the same as one of the named entities
        '''
        
        a = []
        for token in doc:
            # print(str(token) + ' ' + token.pos_)
            if(token.pos_ != "VERB"):
                s = str(token)
                a.append(s)
            else:
                break
        subject = " ".join(a).strip()
        named_entity.add(subject)
        
        return list(named_entity)

ner = NER()


#test
def test():
    #claim = 'Homeland is an American television spy thriller based on the Israeli television series Prisoners of War.'
    #claim = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
    claim = "History of art includes architecture, dance, sculpture, music, painting, poetry literature, theatre, narrative, film, photography and graphic arts."
    #claim = "Charles Marie de La Condamine was born in 1701."
    t1 = time.time()
    e = ner.getNER(claim)
    t2 = time.time()
    print('%.2f' %(t2 - t1))
    print(e)

if __name__ == '__main__':
    test()
