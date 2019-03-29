Transfer Learning NLP
==============================

We will use transfer learning approaches in NLP like CoVe, ELMo, BERT and GPT on Twitter US Airline Dataset.

Project Organization
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
