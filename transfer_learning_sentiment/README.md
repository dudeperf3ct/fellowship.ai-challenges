Transfer Learning NLP
==============================

We will use transfer learning approaches in NLP like CoVe, ELMo, BERT and GPT on Twitter US Airline Dataset.

Blog: [Power of Transfer Learning in NLP](https://dudeperf3ct.github.io/nlp/transfer/learning/2019/02/22/Power-of-Transfer-Learning-in-NLP/)

Feel free to jump anywhere,

- [Approach](#approach)
  - [CoVe](#cove)
  - [ELMo](#elmo)
  - [ULMFiT](#ulmfit)
  - [BERT](#bert)
  - [GPT-2](#gpt-2)
- [Results](#results)
  - [Keras](#keras)
  - [PyTorch](#pytorch)
  - [Fastai](#fastai)
  - [Flair](#flair)
- [Project Organization](#project-organization)

## Approach



### CoVe

For in-depth discussion on Cove, look [here](https://dudeperf3ct.github.io/nlp/transfer/learning/2019/02/22/Power-of-Transfer-Learning-in-NLP/#how-it-works?).

**TL;DR**

- Use the traditional encoder-decoder architecture used in seq2seq learning, to learn the context of words by giving input GLoVe embedding of words in sentence to encoder and two stacked BiLSTM layers generate output is hidden vector or context vectors.
- We looked at one specific example of MT, where encoder was used to generate context vectors, and this context vectors along with attention mechanism (which gives context-adjusted state as output) to give target langauge output sentence using decoder.

### ELMo

For in-depth discussion on ELMo, look [here](https://dudeperf3ct.github.io/nlp/transfer/learning/2019/02/22/Power-of-Transfer-Learning-in-NLP/#how-it-works?).

**TL;DR**

- Different words carry different meaning depending on context and so their embeddings should also take context in account.
- ELMo trains a bidirectional LM, and extract the hidden state of each layer for the input sequence of words.
- Then, compute a weighted sum of those hidden states to obtain an embedding for each word. The weight of each hidden state is task-dependent and is learned.
- This learned ELMo embedding in used in specific downstream tasks for which embedding is obtained.

### ULMFiT

For in-depth discussion on ULMFiT, look [here](https://dudeperf3ct.github.io/nlp/transfer/learning/2019/02/22/Power-of-Transfer-Learning-in-NLP/#how-it-works?).

**TL;DR**

- CV transfer learning style training. Create a pretrained language model by training on large corpus like Wikitext-103, etc.
- Finetune LM data on target data and to stabalize this finetuning two methods like Discriminative finetuning and Slanted learning rates are used.
- To finetune on target task classifier using above finetune LM, additional linear model is added to language model architecture such as concat pooling is added and gradual unfreezing is used.


### BERT

For in-depth discussion on BERT, look [here](https://dudeperf3ct.github.io/nlp/transfer/learning/2019/02/22/Power-of-Transfer-Learning-in-NLP/#how-it-works?).

**TL;DR**

- Use large corpus of unlabeled data to learn a language model(which captures semantics, etc of language) by training on two tasks: Masked Language Model and Next Sentence Prediction using a multi-layer bidirectional Transformer Encoder architecture.
- Finetuning pretrained language model for specific downstream tasks, task-specific modifications are done.


### GPT-2

For in-depth discussion on GPT-2, look [here](https://dudeperf3ct.github.io/nlp/transfer/learning/2019/02/22/Power-of-Transfer-Learning-in-NLP/#how-it-works?).

**TL;DR**

- Large and diverse amount data is enough to capture language semantics related to different tasks instead of training a language model for seperate tasks.
- Pretrained lanaguage model does excellent job on various tasks such as question answering, machine translation, summarization and especially text generation without having to train explicitly for each particular tasks. No task-specific finetuning required.
- GPT-2 achieves mind blowing results just through pretrained language model.

## Result

We will apply the recently learned nlp techniques and see what they can add to the table.

### Keras 

| Approach | Epoch  | Time (sec)  | Train Accuracy(%)  | Dev Accuracy (%)  |
|---|---|:---:|:---:|:---:|
| LSTM  |  10  | 250  |  82 |  80 |
| BiLSTM |  10 |  500 |  83 | 79  |
| GRU  |  10 |  300 |  88 | 77  |
| CoVe  | 10 | 450  | 72  | 72  |
| BERT  |  3 | 500  |  - | 85  |


### PyTorch

|  Approach | Epoch  | Time (sec)  | Train Accuracy(%)  | Dev Accuracy (%)  |
|---|---|:---:|:---:|:---:|
| LSTM  |  10  | 25  |  98 |  78.8 |
| BiLSTM |  10 |  35 |  98 | 79.1  |
| GRU  |  10 |  27 |  92 | 79.3  |
| BERT  |  3 | 600  |  - | 85.03  |

How can I not try open GPT-2 langauge generation model?

Here are the results on random seeds,

Sample Text 1

```
Take me to the QB!

I came pregnant.

"I'm sure kids will be learning that to be a tad boy."<|endoftext|>Embed This Video On Your Site With This Html: Copy Embed code

<iframe src="http://www.youjizz.com/videos/embed/242203" frameborder="0" style="width:100%; height:570px;" scrolling="no" allowtransparency="true"></iframe><|endoftext|>As president-elect Donald Trump takes his dismantling of Barack Obama's health care law to the White House, it's hoped the good will of both generations won't override Kansas' ready-made attorney general. Getting somebody to poke fun at Trump for not keeping things on her from drawing up a repeal budget, following that up with a lame attempt by Sen. Susan Collins, R-Maine, to slash federal penalties on consumers who'd pay more for insurance through a Medicaid replacement, won't be a serious blow.

That may be largely because it works; reaching a single-payer, single-payer healthcare system under Trump's new administration is not much of a magic bullet, and will make us reluctant to defund Planned Parenthood (which is what the Trump administration and the GOP are taking away as a price to pay for shutting it down), but Republicans weigh in as a country trying to find an adequate replacement.

As Chairman of the Joint Committee on Taxation won't the CBO represent, Trump decides, it would give his political opponents less power to make legitimate criticism of AHCA available, or to oppose repeal legislation without sharing the pieces. "Nobody can refuse to make it available. I mean, let's take a look here," said Senate Minority Leader Chuck Schumer, C-N.Y. (four of the sixteen Democratic senators who voted to repeal tax credits for having children care for themselves or their parents, by making free outreach to Russian entertainment hit shows about women injured in Syria). "To me that means that's another piece of legislation from the Oval Office devoted exclusively to people who feel like we're firing them immediately."

Senators are not Republican only because they represent nothing less than sovereignty on their party's national stage, but because they can break through entrenched partisan historical convictions in one moment with one substantive change they want to see voted upon in the next, and then repeat it because that moment feels unreal and basic.

Now this is my heart, because every unelected bit of Republican leadership in the United States has a
```

Sample Text 2

```
[sniffs] I mean Christian, you know, those Christians, we have a holy book that is doctrinal moderate ... like a god? I think that it's time to come out and say like, wow, I just also see the long sweep animosity some people have toward Rule 2. . . but Franciscans also are notoriously violent there and they intend to be in charge at the end of each put. God is the judge of every place."

The Patriot Leader has a tone similar to that of Rev. Carl Nelson's."Let me ask, is a right to exhibit racial and religious symbols on your property such a my God," he marveled, as one inquirer characterized those religious symbols."Jeb Talmage, you did this painting there, did you know?" asked Gagnon.

Through a crack at Trump for his response, Fitzgerald efficient rebuked; not once did Fitzgerald otherwise react. More snd just finished writing this post which the Journal overlooked due to focus and questionability and yet in this city transcribed the posting over 887 frustrating seconds in length and markedly. You know probably will not boo the level of the Presidential "say turn" for much money to basically pay for his utter disregard for our how we live. Not that there's anything wrong with pay-to-play or kicking Fauxouts. As a mother of one young daughter, pray for my voice. See "Before this she left to go to Gothic, I looked up the alphabet to eat cookies For my Christian a Werewolf Wolf, and my Dog ; quote: Heaven my prayer

I saw the scene these moronchildren were living in,

they had room to spare from their loves.

I couldn't make any conversation like a Walter White subject... Reject madmen of the City of Independence. [[End Post, 5/17/08] Hearing what Fitzgerald Barbara is saying, began to seem superfluous. Fitzgerald burst into a Googling of white people's black "hypocrisy" and found herself empowered with an insight understanding that the principle of "black roots arrogant of white" must be Austin McKetty's's vision of white supremacy, when it was the preeminence of a man and its map to totalitarianism, its sensibilities, its pleading what could be called "impartiality to all our problems" and the above alienation of white people. Within and of the Gentlemen's splashy rendition of the words, The Advocate adapted this core reading to help these white nationalist bast
```

*Mind blowing* 🤯

### Fastai

|  Approach | Epoch  | Time (min)  | Train loss |  Dev loss | Dev Accuracy (%)  |
|---|---|:---:|:---:|:---:|
| Finetune LM | 15   |  6 | 3.575478 | 4.021957  | 26.4607 |
| Finetune Classifier | 5   |  2 | 0.786838  |	0.658620  | 72.4479 |
| Gradual Unfreezing (Last 1 layer) | 5   |  2 | 0.725324  |	0.590953  | 75.2134 |
| Gradual Unfreezing (Last 2 layer) | 5   |  3 | 0.556359  |	0.486604   | 81.2564 |
| Unfreeze whole and train | 8   |  7 |  0.474538  |	0.446159  | 82.9293 |

### Flair

#### ELMo

negative   tp: 854 - fp: 124 - fn: 106 - tn: 380 - precision: 0.8732 - recall: 0.8896 - accuracy: 0.7878 - f1-score: 0.8813

neutral    tp: 141 - fp: 79 - fn: 141 - tn: 1103 - precision: 0.6409 - recall: 0.5000 - accuracy: 0.3906 - f1-score: 0.5617

positive   tp: 170 - fp: 96 - fn: 52 - tn: 1146 - precision: 0.6391 - recall: 0.7658 - accuracy: 0.5346 - f1-score: 0.6967



#### BERT

tp: 814 - fp: 97 - fn: 146 - tn: 407 - precision: 0.8935 - recall: 0.8479 - accuracy: 0.7701 - f1-score: 0.8701

neutral    tp: 188 - fp: 157 - fn: 94 - tn: 1025 - precision: 0.5449 - recall: 0.6667 - accuracy: 0.4282 - f1-score: 0.5997

positive   tp: 159 - fp: 49 - fn: 63 - tn: 1193 - precision: 0.7644 - recall: 0.7162 - accuracy: 0.5867 - f1-score: 0.7395


#### Flair

negative   tp: 854 - fp: 193 - fn: 106 - tn: 311 - precision: 0.8157 - recall: 0.8896 - accuracy: 0.7407 - f1-score: 0.8510

neutral    tp: 138 - fp: 112 - fn: 144 - tn: 1070 - precision: 0.5520 - recall: 0.4894 - accuracy: 0.3503 - f1-score: 0.5188

positive   tp: 121 - fp: 46 - fn: 101 - tn: 1196 - precision: 0.7246 - recall: 0.5450 - accuracy: 0.4515 - f1-score: 0.6221



#### ELMo + BERT

negative   tp: 838 - fp: 91 - fn: 122 - tn: 413 - precision: 0.9020 - recall: 0.8729 - accuracy: 0.7973 - f1-score: 0.8872

neutral    tp: 152 - fp: 81 - fn: 130 - tn: 1101 - precision: 0.6524 - recall: 0.5390 - accuracy: 0.4187 - f1-score: 0.5903

positive   tp: 199 - fp: 103 - fn: 23 - tn: 1139 - precision: 0.6589 - recall: 0.8964 - accuracy: 0.6123 - f1-score: 0.7595



#### Flair + BERT


negative   tp: 895 - fp: 158 - fn: 65 - tn: 346 - precision: 0.8500 - recall: 0.9323 - accuracy: 0.8005 - f1-score: 0.8892

neutral    tp: 134 - fp: 64 - fn: 148 - tn: 1118 - precision: 0.6768 - recall: 0.4752 - accuracy: 0.3873 - f1-score: 0.5584

positive   tp: 166 - fp: 47 - fn: 56 - tn: 1195 - precision: 0.7793 - recall: 0.7477 - accuracy: 0.6171 - f1-score: 0.7632



#### Flair + ELMo

negative   tp: 827 - fp: 86 - fn: 133 - tn: 418 - precision: 0.9058 - recall: 0.8615 - accuracy: 0.7906 - f1-score: 0.8831

neutral    tp: 182 - fp: 116 - fn: 100 - tn: 1066 - precision: 0.6107 - recall: 0.6454 - accuracy: 0.4573 - f1-score: 0.6276

positive   tp: 182 - fp: 71 - fn: 40 - tn: 1171 - precision: 0.7194 - recall: 0.8198 - accuracy: 0.6212 - f1-score: 0.7663


---


## Project Organization

------------

        .
        ├── data
        ├── docs
        │   ├── commands.rst
        │   ├── conf.py
        │   ├── getting-started.rst
        │   ├── index.rst
        │   ├── make.bat
        │   └── Makefile
        ├── LICENSE
        ├── Makefile
        ├── models
        ├── notebooks
        │   ├── tl_nlp_sentiment_allennlp.ipynb       <-- Jupyter notebook with ELMo and BERT using allenlp framework
        │   ├── tl_nlp_sentiment_fastai.ipynb         <-- Jupyter notebook with ULMFiT using fastai framework
        │   ├── tl_nlp_sentiment_flair.ipynb          <-- Jupyter notebook with ELMo, BERT and Flair using flair framework
        │   ├── tl_nlp_sentiment_keras.ipynb          <-- Jupyter notebook with CoVe and BERT using keras framework
        │   └── tl_nlp_sentiment_pytorch.ipynb        <-- Jupyter notebook with BERT and GPT-2 using pytorch framework
        ├── README.md
        ├── requirements.txt
        ├── setup.py
        ├── src
        │   ├── data
        │   │   ├── README.md
        │   ├── features
        │   │   ├── build_features.py
        │   │   └── __init__.py
        │   ├── __init__.py
        │   ├── models
        │   │   ├── __init__.py
        │   │   ├── predict_model.py
        │   │   └── train_model.py
        │   └── visualization
        │       ├── __init__.py
        │       └── visualize.py
        ├── test_environment.py
        └── tox.ini


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
