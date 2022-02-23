from data import config
from build_dataloader import corpus
import torch
import string
import pandas as pd
from tqdm import tqdm



def infer_test(checkpoint_path, sentence, true_tags=None):

    model = torch.jit.load(checkpoint_path)
    model.eval()
    tokens = sentence
    numericalized_tokens = [corpus.word_field.vocab.stoi[t] for t in tokens]
    # find unknown words
    unk_idx = corpus.word_field.vocab.stoi[corpus.word_field.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
    token_tensor = torch.LongTensor(numericalized_tokens)
    token_tensor = token_tensor.unsqueeze(-1)
    predictions = model(token_tensor)
    top_predictions = predictions.argmax(-1)
    predicted_tags = [corpus.tag_field.vocab.itos[t.item()] for t in top_predictions]

    return tokens, predicted_tags, unks

def main():

    checkpoint_path = config.CHECKPOINT3

    sentences=[]
    tags = []

    for sentence in tqdm(corpus.test_dataset):
        words, infer_tags, unknown_tokens = infer_test(checkpoint_path, sentence = sentence.word, true_tags=sentence.tag)
        sentences.append(words)
        tags.append(infer_tags)

    print(len(sentences))
    print(len(tags))

    data=[]


    for i,(sen,tag) in enumerate(zip(sentences, tags)):
        name = []
        loc = []
        org = []

        sentence = ["".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()]

        for j, ent in enumerate(tag):
            if ent == 'B-PER':
                name.append(sen[j])

            elif ent == 'I-PER':
                name.append(sen[j])

            elif ent == 'B-LOC':
                loc.append(sen[j])

            elif ent == 'I-LOC':
                loc.append(sen[j])

            elif ent == 'I-ORG':
                org.append(sen[j])

            elif ent == 'B-ORG':
                org.append(sen[j])

        data.append([i+1, sentence, name, loc, org])

    print(data[:5])

    df = pd.DataFrame(data, columns = ['Sl', 'Text', 'Extracted Name', 'Extracted Location', 'Extracted Organization'])


    df.to_csv('output/output.csv', index=False)
    print(df.head(5))




if __name__=='__main__':
    main()