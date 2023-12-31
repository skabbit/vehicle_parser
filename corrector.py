import os
import pathlib
import pickle
import re
import json
import pandas as pd
from collections import defaultdict
from symspellpy import SymSpell
from symspellpy.editdistance import DistanceAlgorithm
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tag import UnigramTagger
from nltk.tokenize import LineTokenizer, WordPunctTokenizer
from tqdm import tqdm
from weblib.russian import slugify

PATH_TO_FILE = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


class CarBrandManager():
    def load(self):
        df = pd.read_csv(PATH_TO_FILE / "carbrand.csv")
        df = df.set_index('id')
        self.brands = df.to_dict(orient="index")
        
        df = pd.read_csv(PATH_TO_FILE / "alt_names.csv")
        df = df[['name', 'carbrand_id']]
        self.alt_names = dict(df.values.tolist())
   
    def get(self, id: int):
        if id not in self.brands:
            return None
        return CarBrand(d=self.brands[id])


class CarBrand():
    objects = CarBrandManager()

    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
        self.parent = d['parent_id']

    def __str__(self):
        if self.parent_name:
            return '%s %s' % (self.parent_name, self.name)

        return self.name

    @staticmethod
    def get_mark_name_pairs():
        return CarBrand.objects.alt_names

    @staticmethod
    def slugify(name, parent_name=None):
        original_name = name
        name = name.replace("_", " ")

        name_pairs = CarBrand.get_mark_name_pairs()
        for a in name_pairs:
            if slugify(name) == slugify(a):
                name = name_pairs[a]
                break

        if parent_name:
            name = name.replace("(%s)" % parent_name, "").strip()

        name = re.sub(r"/[^\s]+/", r"", name)
        name = name.replace("π", "pi")
        name = name.replace("-Class", "")

        name = slugify(name)
        name = name.replace("-", "")
        name = name.replace("series", "").replace("seriya", "").replace("serie", "")
        name = name.replace("klasse", "").replace("klass", "")

        if name.startswith("model"):
            name = name[5:]

        if parent_name:
            # cut brand from model name
            parent_name = CarBrand.slugify(parent_name)

            if name.startswith(parent_name):
                name = name[len(parent_name):]
            if name.endswith(parent_name):
                name = name[:-len(parent_name)]

            if parent_name in ["vaz", "uaz"]:
                if re.match(r"^\d+.+", name):
                    name = re.sub(r"^(\d+)[^0-9].*", r"\1", name)

            if parent_name == "gaz":
                if len(name) > 5:
                    name = re.sub(r"wolga$", r"", name)

            if parent_name == "bmw":
                name = re.sub(r"^(\d)er(\d+)", r"\2", name)
                name = re.sub(r"^(\d)er", r"\1", name)
                name = re.sub(r"^(\d)serii(\d+)", r"\2", name)
                name = re.sub(r"^(\d)serii", r"\1", name)
                name = name.replace("granturismo", "gt").replace("grancoupe", "gc")

            if parent_name == "alfaromeo":
                name = re.sub(r"^alfa", r"", name)

            if parent_name == "mercedesbenz":
                name = re.sub(r"^mercedes", r"", name)

            if parent_name == "moskvich":
                name = re.sub(r"^moskwitsch", r"", name)

            if parent_name == "volkswagen":
                name = re.sub(r"^vw", r"", name)

        # in case of empty slug (eg: "Mini Mini" by Mini), just keep as it was
        if not name:
            name = slugify(original_name).replace("-", "")

        return name    

class LevenshteinSymSpell(SymSpell):
    def __init__(self,
                 max_dictionary_edit_distance=2,
                 prefix_length=7,
                 count_threshold=1):
        super(LevenshteinSymSpell, self).__init__(max_dictionary_edit_distance, prefix_length, count_threshold)
        self._distance_algorithm = DistanceAlgorithm.DAMERUAUOSA


class CarBrandCorrector:
    def __init__(self,
                 path=PATH_TO_FILE,
                 use_freqs=True,
                 prepare_chars="replace",  # regex | replace | none
                 use_tagger=False,
                 split_text=True,
                 try_model_name_only=False,  # without mark name
                 disable_tqdm=True,
                 word_segmentation_strategy="lookup_compound",
                 max_dictionary_edit_distance=0):
        path = pathlib.Path(path)
        self.path = path
        self.prepare_chars = prepare_chars
        self.use_tagger = use_tagger
        self.split_text = split_text
        self.disable_tqdm = disable_tqdm
        self.try_model_name_only = try_model_name_only
        self.word_segmentation_strategy = word_segmentation_strategy
        self.max_dictionary_edit_distance = max_dictionary_edit_distance

        dictionary_file = "dictionary.txt"
        self.dictionary_file = path / dictionary_file
        self.freqs_file = path / "freqs.txt"
        self.bigram_file = path / "bigrams.txt"
        self.corrector_file = path / "corrector.txt"
        self.tokenizer = WordPunctTokenizer()
        self.word_freqs = defaultdict(int)
        self.bigram_freq = defaultdict(int)

        self.reader = TaggedCorpusReader(str(path), dictionary_file, word_tokenizer=LineTokenizer(), sep='/')

        if use_tagger:
            self.tagger = UnigramTagger(self.reader.tagged_sents())

        if prepare_chars in ['regex', 'combine']:
            self.layout_regex_list = pickle.load(open(path / "wrong_layout_regex.pickle", "rb"))

            for name in self.layout_regex_list.keys():
                self.layout_regex_list[name] = re.compile(self.layout_regex_list[name], flags=re.U)

        # one dict to replace tagger
        fp = open(self.dictionary_file, encoding="utf-8")
        self.cardict = {}
        for line in fp.readlines():
            name, pk = line.strip().split("/")
            self.cardict[name] = int(pk)
        fp.close()

        # SymSpell
        self.symspell = SymSpell()
        if use_freqs:
            self.symspell.load_dictionary(self.freqs_file, 0, 1)
            self.symspell.load_bigram_dictionary(self.bigram_file, 0, 2)
        else:
            corpus_path = path / "corrector.txt"
            self.symspell.create_dictionary(str(corpus_path))

    def get_last_entity_model(self, text):
        doc = self.splitter.nlp(text.strip())
        for entity in reversed(doc.ents):
            if entity.label_ == "MODEL":
                return entity.text

    def tokenize(self, name):
        result = []
        for label, text in self.splitter.nlp.tokenizer.explain(name):
            if label == "TOKEN":
                result.append(text)
        return result

    def add_name_freq(self, name, freq):
        words = name.split(" ")
        for word in words:
            self.word_freqs[word] += freq

        if len(words) > 1:
            last_word = words[0]
            for word in words[1:]:
                self.bigram_freq["%s %s" % (last_word, word)] += freq
                last_word = word

    def print_model_name(self, text):
        model = self.detect_model(text)
        if model:
            return str(CarBrand.objects.get(id=model))

    def detect_model(self, text: str):
        """
        :param text:
        :return: int (model_id) or None
        """
        result = self.detect_model_by_text(text)
        if not result or (result and not CarBrand.objects.get(id=result).parent):
            if result:
                obj = CarBrand.objects.get(id=result)
            slug_result = self.detect_model_by_text(CarBrand.slugify(text))
            if slug_result:
                if not result:
                    return slug_result
                obj2 = CarBrand.objects.get(id=slug_result)
                if obj2.parent == obj:
                    return slug_result

        return result

    def get_tag(self, token):
        if self.use_tagger:
            pk = self.tagger.tag([token])[0][1]
        else:
            pk = self.cardict.get(token)
        return pk

    def detect_model_by_text(self, text: str):
        if self.split_text:
            model_text = self.get_last_entity_model(text)
            if model_text:
                text = model_text

        text = self.prepare_model_text(text)

        # correct spelling + word split by SymSpell
        if self.word_segmentation_strategy == "word_segmentation":
            text = self.symspell.word_segmentation(text,
                                                   max_edit_distance=self.max_dictionary_edit_distance).segmented_string
        elif self.word_segmentation_strategy == "lookup_compound":
            suggestions = self.symspell.lookup_compound(text.strip(), max_edit_distance=2)
            text = suggestions[0].term
        elif self.word_segmentation_strategy == "both":
            text = self.symspell.word_segmentation(text,
                                                   max_edit_distance=self.max_dictionary_edit_distance).segmented_string
            suggestions = self.symspell.lookup_compound(text.strip(), max_edit_distance=2)
            text = suggestions[0].term

        # tokenize
        # tokens = self.tokenize(text)
        tokens = [text]

        # join string token-by-token while has related existing tag
        spaced_name = ''
        joint_name = ''
        result = None
        tokens_passed = set()
        for token in tokens:
            # "Maybach МАЙБАХ 57" или "ВАЗ VAZ-2107"
            token_tag = self.get_tag(token)
            if token_tag and int(token_tag) == result:
                continue

            if token in tokens_passed and token != "rover":
                continue

            spaced_name += token
            joint_name += token
            tokens_passed.add(token)

            model = self.get_tag(spaced_name)
            if model:
                result = int(model)
            else:
                model = self.get_tag(joint_name)
                if model:
                    result = int(model)
                else:
                    model = self.get_tag(CarBrand.slugify(joint_name))
                    if model:
                        result = int(model)

            spaced_name += " "

        return result

    CYR_TO_ENG_LAYOUT = {
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "Н": "H",
        "К": "K",
        "М": "M",
        "О": "O",
        "о": "o",
        "Р": "P",
        "Т": "T",
        "Х": "X",
        "У": "Y",
    }

    def prepare_model_text(self, text):
        # tokenize -> replace cyrillic a/e/t/c to latin letters -> lowercase
        tokens = [text]
        text = ""
        wrong_chars = "".join(self.CYR_TO_ENG_LAYOUT.keys())

        # if first token is digits only (GIBDD frequent case) - skip it
        if len(tokens) > 1 and re.match(r"\d+$", tokens[0]):
            tokens = tokens[1:]

        for token in tokens:
            if self.prepare_chars == "replace":
                if re.match(r"[a-zA-Z0-9\-%s]+$" % wrong_chars, token):
                    for char, new_char in self.CYR_TO_ENG_LAYOUT.items():
                        token = token.replace(char, new_char)
            elif self.prepare_chars == "regex":
                for name, pattern in self.layout_regex_list.items():
                    if pattern.match(token):
                        token = name
            else:
                if len(re.findall(r"[a-zA-Z0-9]", token)) > len(re.findall(r"[а-яёА-ЯЁ]", token)):
                    for char, new_char in self.CYR_TO_ENG_LAYOUT.items():
                        token = token.replace(char, new_char)

            text += token + " "

        text = text.lower()
        return text


CarBrand.objects.load()
corrector = CarBrandCorrector(path=PATH_TO_FILE / "config" / "stable", split_text=False)

if __name__ == "__main__":
    model_id = corrector.detect_model("ваз 2107")
    print(model_id)
    print(CarBrand.objects.get(id=model_id))

    df = pd.read_csv("car_test.csv")
    df['carbrand_id'] = df.apply(lambda row: corrector.detect_model(row['marka']), axis=1)
    df