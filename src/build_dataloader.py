from data.dataloader import Corpus



val = int(input("Want to load embedding from pretrained word2vec?(0/1):"))

if val == 0:
    corpus = Corpus(
        input_folder="dataset/",
        train_data= "conllpp_up_train.tsv",
        val_data="conllpp_up_dev.tsv",
        test_data="conllpp_up_test.tsv",
        min_word_freq=3,
        batch_size=64,
    )
else:

    corpus = Corpus(
        input_folder="dataset/",
        train_data= "conllpp_up_train.tsv",
        val_data="conllpp_up_dev.tsv",
        test_data="conllpp_up_test.tsv",
        min_word_freq=3,
        batch_size=64,
        w2v=1
    )



train_iter = corpus.train_iter
val_iter = corpus.val_iter
test_iter = corpus.test_iter


print(f"No of sentences in Train set: {len(corpus.train_dataset)} sentences")
print(f"No of sentences in Val set: {len(corpus.val_dataset)} sentences")
print(f"No of sentences in Test set: {len(corpus.test_dataset)} sentences")


print("\nNo of batches in train dataset:", len(train_iter))
print("No of batches in validation dataset:", len(val_iter))
print("No of batches in test dataset:", len(test_iter))