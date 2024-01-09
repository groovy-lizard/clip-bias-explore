from nltk.corpus import wordnet as wn
import argparse


class Allonyms:
    def __init__(self, word):
        self.word = word
        self.to_string = str()
        self.synms = set()
        self.simls = set()
        self.sees = set()
        self.hypos = set()
        self.hypers = set()
        self.allonyms = set()

    def update_word(self, word):
        self.word = word
        return self.word

    def grab_synms(self):
        for syn in wn.synsets(self.word):
            for lem in syn.lemmas():
                self.synms.add(lem.name())
        return self.synms

    def grab_sees(self):
        for syn in wn.synsets(self.word):
            for als in syn.also_sees():
                for lem in als.lemmas():
                    self.sees.add(lem.name())
        return self.sees

    def grab_simls(self):
        for syn in wn.synsets(self.word):
            for sims in syn.similar_tos():
                for lem in sims.lemmas():
                    self.simls.add(lem.name())
        return self.simls

    def grab_hypos(self):
        for syn in wn.synsets(self.word):
            for hypo in syn.hyponyms():
                for lem in hypo.lemmas():
                    self.hypos.add(lem.name())
        return self.hypos

    def grab_hypers(self):
        for syn in wn.synsets(self.word):
            for hype in syn.hypernyms():
                for lem in hype.lemmas():
                    self.hypers.add(lem.name())
        return self.hypers

    def grab_all(self):
        synms = self.grab_synms()
        sees = self.grab_sees()
        simls = self.grab_simls()
        hypos = self.grab_hypos()
        hypers = self.grab_hypers()
        self.allonyms = synms.union(sees, simls, hypos, hypers)
        return self.allonyms

    def parse_string(self):
        self.to_string += "{"
        self.to_string += f"'{self.word}',\n"
        for name in self.allonyms:
            self.to_string += f"'{name}',\n"
        self.to_string += "}\n"
        return self.to_string

    def to_file(self, filename):
        f = open(filename, 'a')
        f.write(self.to_string)
        f.close()
        return "ok"

    def all_to_file(self, filename):
        self.grab_all()
        self.parse_string()
        self.to_file(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Allonyms",
        description="Retrieve all kinds of 'nyms' of a given word"
    )
    parser.add_argument('word')
    parser.add_argument('-f', '--file', default='race-allonyms.txt')
    args = parser.parse_args()
    allo = Allonyms(args.word)
    allo.all_to_file(args.file)
