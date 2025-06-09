# Summer 2025 AI Week : Topic Modeling to Categorize Text Documents 

*September 3, 2025*

Do you need to identify a set of themes from within a large collection of text documents?  If so, topic modeling can help.  For example, topic modeling can be used to identify recurring themes in news articles, discover research trends in scientific publications, or analyze public sentiment across social media posts. In this workshop, you will get a high-level overview of different existing techniques used for topic modeling which focuses on modern AI-driven approaches. You will also have ample time to work through step-by-step hands-on exercises to learn how to leverage AI-based topic modeling analysis techniques using real data through Python. While previous knowledge of machine learning / text analysis packages in Python is not strictly required, familiarity with the Python programming language is encouraged. 


## Links to workshop materials on Colab

- [BERTopic demo notebook](https://colab.research.google.com/github/nuitrcs/AI_Week_Topic_Model/blob/main/BERTopic-demo.ipynb)

- [Exercise 1 notebook](https://colab.research.google.com/github/nuitrcs/AI_Week_Topic_Model/blob/main/exercises/exercise1.ipynb)


## If you want to work on your local computer 

If you are working on your local computer, we recommend that you setup of a conda environment for this workshop.  You can use the command below to do so:

```
conda create --name topic-modeling-env -c conda-forge python=3.12 jupyter matplotlib pandas numpy=2.2 scikit-learn sentence-transformers umap-learn hdbscan bertopic datashader bokeh holoviews scikit-image colorcet keybert

```

Then you can download (or clone) this repo and use the provided .ipynb files (running in your topic-modeling-env conda environment).