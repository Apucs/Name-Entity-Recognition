# Name Entity Extraction

## __Introduction:__  

   In the Project, we are about to perform Name Entity Extraction using the custom build neural model.
   > The dataset that we have used is called **CoNLL2003++** dataset. This dataset is created by **Wang et al.** at 2020, the **CoNLL 2003 ++** Similar to the original **CoNLL** except test set has been corrected for label mistakes. The dataset is split into training, development, and test sets, with 14,041, 3,250, and 3,453 instances respectively., in English language. Containing 20,744 in Text file format.  
   > Dataset can be downloaded from the given [Link](https://metatext.io/datasets/conll-2003-++).


## __Project Description__

Root directory of this project contains:
>
> - **4 sub-folders**
> - **several [.py] files**
> - **1 text files containing all the requirements of this project**
> - **1 readme.md file containing all the necessary instructions.**
> - **Some test image examples**

Details about the folders and files:
 >
 > - **checkpoint(folder):**  Contains the checkpoint and trained model which will be used to test or evaluate our model.  
 > - **dataset(folder):** Contains out **train, test and validation** data.

 > - **output(folder):** Contains all the output that we got from the trained model. Outputs contain-
 >
 >> - **output.csv** : Output from the base model
 >> - **output_w2v** : Output from the model trained with pretrained **word2vec** model.
 >> - **result.csv** : File contains extracted required entites acquired from the **base model**.
 >> - **result_w2v.csv**: File contains extracted required entites acquired from the **word2vec model**.
 >> - **images**:  some images indicates our model's loss and accuracy. 

 > - **src(folder)**: Contains all the source codes for this project. This folder contains:  
 >
 >> - **data(package)**: Contains the following files:
 >>
 >>> - **\_\_init__.py**
 >>> - **config.py**: Our initial file and path configuration for this project.
 >>> - **dataloader.py**: This script will be used to create the vocabulary with our dataset and build our dataloader.
 >>> - **processor.py**: Before build our dataloader we need to process our dataset and make the right format for the desired output so that this dataset can be used to train with.

 >> - **models(package)**: Contains the following files:
 >>
 >>> - **\_\_init__.py**
 >>> - **model_w2v.py**: This model will be used if we want to train the model with pretrained **Word2Vec** embedding vectors.
 >>> - **model.py**: This model is for train the model with random embedding.
 >>> - **NER.py**: This script will be used for providing us with the different helping functionalities.
 >>>

 >> - **rnd**: Contains `bert.ipynb` file.
 >>>
 >>> - I also did some experiments with pretrained `bert-based-cased` model with our data.

 >> - **\_\_init__.py**  
 >> - **build_dataloader.py**: Will be used to build our dataloader.
 >> - **processing.py**: Will be used to process our data.
 >> - **train.py**: Will be used to initialize and train our model.
 >> - **evaluate.py**: Evaluate our model with test data.
 >> - **test.py**: It will be used to test our model with our test data and generate the `output.csv`  file by extracting our desired entities for each given test sentences.
 >> - **inference.py**: We can test our model for a single given sentence.

### __Project intution:__

> In this project we are about to extract name entity from the model that we are about to build. There are several ways to build or use the models to get state of the art performance. Among those models most popular approches are:
>
> > - **embedding + LSTM +CNN**
> > - **embedding + Bi-LSTM + CNN**
> > - **Pretrained Embedding(Word2Vec/GloVe) + Bi-LSTM + CNN**
> >
> > - **Embedding + Bi-LSTM + CNN + CRF**
> > - Fine tuning pretrained **NER** models(**BERT**)

> In this project, we have tried to build the model using **2nd** and **3rd** approach from the above list and tried to get state of the art result.
  
>
### __Instructions:__

> Before starting we need to satisfy all the dependencies. For that reason need to execute the following command. (All the commands need to be executed from the root folder)
>
> - __Install the dependencies:__
    `pip install -r requirements.txt`  

> Before start the training we need to process our dataset into the correct format. To do that execute the following command:
>
> - __To process the dataset:__
    `python src/processing.py`  

> Now it's time to start training our model. There are two models that we can use to train.
> >
> > - Without pretrained Word2Vec embedding
> > - With pretrained Word2Vec embedding 
>  
> There are also some hyperparameters that we can set during training.
> >
> > - Embedding dimension we want
> > - Hidden dimension
> > - No. of Bi-LSTM layers
> > - Embedding dropout
> > - Bi-LSTM dropout
> > - FC layer dropout
> > - No. of epoch
> > - If we want to train with word2vec or not(1/0).

> To start the training-
 >
 > - __To train without pretrained word2vec model__  
  `python src/train.py --emd 300, --hidden 128, --lstm 2, --em_drop 0.30, --lstm_drop 0.20, --fc_drop 0.30, --epoch 10`  

 > After executing this command there will be a question if we want to load embedding from pretrained word2vec. We need to set it to `0`.

 > - __To train with pretrained word2vec model__  
  `python src/train.py --emd 300, --hidden 128, --lstm 2, --em_drop 0.30, --lstm_drop 0.20, --fc_drop 0.30, --epoch 10 --w2v 1`  

> After executing this command there will be a question if we want to load embedding from pretrained word2vec. We need to set it to `1`.

> - __To evaluate the trained model:__  
> `python src/evaluate.py`

> If our model was not trained on pretrained word2vec, We always need to set the value to `0` otherwise `1`.  
>
> To extract the entities from the test sentences to the **csv** file, following command need to be executed-
>
> - __To extract the entities from the test set:__  
> `python src/test.py`  
>
> - __Test the model with a single sentence:__  
> `python src/inference.py`
>

## Inference

> For our test data, we got the accuracy of -
> >
> > - 94.76%(without pretrained word2vec)
> > - 93.71(with pretrained word2vec)


__Without using word2vec:__

![Training graph](.\output\train.png)

![validation graph](.\output\val.png)

__With using word2vec:__

![Training graph](.\output\train_w2v.png)

![validation graph](.\output\val_w2v.png)
