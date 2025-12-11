# Angela Merrill COMP 331 Final: Grammar Correction 
Grammar processing is a more complex problem for ANNs than simple sentiment analysis. The machine cannot simply associate words with their meanings, it must also learn which words can be combined in which way. I decided to use this concept to fine-tune two models (classification and generation) in order to create a grammar proofreader. I used McCormick and Ryan’s “BERT Fine-Tuning Tutorial with PyTorch” tutorial as well as Professor Zhang’s “GPT-2 Fine-Tuning Tutorial with PyTorch & Huggingface in Colab” as guides on how to set up the modules and structure the data. When provided text, each sentence is classified as grammatical or not by a BertForSequenceClassification model. Ungrammatical sentences are sent to the GPT-2 to generate more grammatically correct versions. 

## How to run: 
Because of the computing requirements, I recommend running the Google Colab notebook.
