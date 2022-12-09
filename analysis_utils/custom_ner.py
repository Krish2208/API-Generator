import spacy
from spacy.tokens import DocBin
import json


nlp = spacy.blank("en")
doc_bin = DocBin()

with open('./data/clean_data/complete.json', 'r') as f:
    train_data = json.load(f)

i = 0
for train_example in train_data:
    text = train_example['text']
    labels = train_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skip", i)
            i+=1
        else:
            ents.append(span)
    filtered_ents = spacy.util.filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("train_ammw.spacy")