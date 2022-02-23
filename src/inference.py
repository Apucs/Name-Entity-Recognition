import torch
from spacy.lang.en import English
from build_dataloader import corpus
from data import config



def infer(checkpoint_path, sentence, true_tags=None):

    model = torch.jit.load(checkpoint_path)
    model.eval()
    # tokenize sentence
    nlp = English()
    tokens = [token.text.lower() for token in nlp(sentence)]
    print("\n",tokens)
    # transform to indices based on corpus vocab
    numericalized_tokens = [corpus.word_field.vocab.stoi[t] for t in tokens]
    # find unknown words
    unk_idx = corpus.word_field.vocab.stoi[corpus.word_field.unk_token]
    unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
   
    print("Tokens size:", len(tokens))
    token_tensor = torch.LongTensor(numericalized_tokens)
    print("Tokens size long Tensor:", token_tensor.shape)
    token_tensor = token_tensor.unsqueeze(-1)
    print("Tokens size updated:", token_tensor.shape)
    predictions = model(token_tensor)
    print("Size of the predictions:", predictions.size())
    top_predictions = predictions.argmax(-1)
    print("Size of the top predictions:", top_predictions.size())
    predicted_tags = [corpus.tag_field.vocab.itos[t.item()] for t in top_predictions]
    
    modified_tags = predicted_tags

    #print(corpus.tag_field.vocab.itos)  
    ###['<pad>', 'O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-LOC', 'I-MISC']

    for i, tag in enumerate(modified_tags):
        if tag == "I-PER" or tag == "B-PER":
            modified_tags[i] = "PERSON"

        elif tag == "I-ORG" or tag == "B-ORG":
            modified_tags[i] = "ORGANIZATION"

        elif tag == "I-LOC" or tag == "B-LOC":
            modified_tags[i] = "LOCATION"

        else:
            modified_tags[i] = "O"


    print("\n")
    print("word".ljust(20), "entity")
    print("-".ljust(30,"-"))



    for word, tag in zip(tokens, modified_tags):
        print(word.ljust(20), tag)



    return tokens, predicted_tags, modified_tags, unks 


def main():
    checkpoint_path = config.CHECKPOINT3

    sen = "Mark Elliot Zuckerberg is an American internet entrepreneur. He is known for co-founding the social media website Facebook and its parent company Meta, located in Menlo Park, California"

    words, infer_tags, mod_infer_tags, unknown_tokens = infer(checkpoint_path, sen, true_tags=None)

    print("Unknow tokens:", unknown_tokens)

if __name__=='__main__':
    main()
