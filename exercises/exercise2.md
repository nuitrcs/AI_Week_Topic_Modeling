# Exercise 2: Applying Topic Modeling to New Text Data

In this exercise, you'll apply the topic modeling workflow to a new dataset of your choice. The goal is to deepen your understanding and get hands-on experience on the process. We encourage you to explore different models and parameter settings to find interesting patterns in the text data.



## Step 1: Choose Your Dataset

You have a few options for selecting your text data:

- Preloaded datasets: Several datasets are prepared and available for you in the `data/` folder in this repo. If you decide to use one of these datasets, you can read in the dataset using the `pandas` Python package. We also found some datasets on huggingface that you can easily retrieve to your system via the huggingface API. Examples for reading in the preloaded datasets is provided at the end of this document.

- Custom dataset: You are also welcome to find and use your own text dataset.

**Tip:** Try to choose a dataset with enough textual content (at least several hundred documents)



## Step 2: Review and Apply the Topic Modeling Workflow

Use the slides, demo code, and insights from Exercise 1 to guide your implementation. Your main steps include:

1. **Embedding**: Use a SentenceTransformer model to convert your documents into vector representations. Try comparing models like `all-MiniLM-L6-v2`, `paraphrase-MPNet`, etc.
2. **Dimensionality Reduction**: Apply techniques such as UMAP to reduce the embedding dimensionality. Tune parameters like `n_neighbors` or `min_dist` to get optimal results.
3. **Clustering**: Cluster the reduced embeddings using HDBSCAN or other clustering algorithms and change parameters as neede to retrieve the best clustering.
4. **Representation**: Generate topic representations using BERTopic or other methods. Adjust parameters like the number of top words or the vectorizer settings.



## Step 3: Explore and Interpret Topics

- Examine your topics. Are they coherent?
- Are there obvious or surprising themes?
- Remember that you may have to iterate many times, changing parameters for many individual steps in the workflow in order to retrieve your best set of topics.  We recommend changing one parameter at a time, checking how this effects your results, then moving on to another parameter.



## Step 4: Bonus – Try BERTopic Variations

Once you're comfortable with the basic setup and have found the best set of topics you can for your dataset, push further by exploring advanced features in BERTopic. For example:

- **Dynamic Topic Modeling**: Visualize how topic prevalence changes over time ([docs](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html)).  This may be an interesting option of your dataset has a column for time.
- **Hierarchical Topic Modeling**: Create a visualization that lets you see the hierarchical nature of your topics. You can use this to decide if some topics should be merged together ([docs](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html)).


## Deliverables

At the end of the day, we would like you to share your work with others. Below are examples of the materials that you could prepare to demonstrate your approach and what interesting findings you made during the exercise.  (You do not have to produce all of these materials.)

- A brief write-up or notebook summarizing your topic modeling process.
- Visualizations: topic distributions, top words per topic, or topic evolution plots.
- Notes on interesting or unexpected insights from your dataset.



## 💡 Tips

- Preprocessing is often an important step. Explore the raw text data first and check to see if there are texts that are empty, full of blanks, or too short that it doesn't have any meaningful content.  If so, you can write code (like in our example notebooks) to remove rows of your dataframe that don't have meaningful content.
- Topic modeling is often exploratory — it's okay to not find perfect topics on the first try!
- Use visual tools (like `.visualize_topics()` in BERTopic) to understand your results.

**Good luck! And don’t hesitate to ask questions or collaborate with others as you go.**



## Datasets

### Laws Dataset ([source](https://enjalot.github.io/latent-scope/us-federal-laws))

If you are working on a notebook in the `exercises/` directory and you want to use the `laws_topic_model.csv` dataset, you can run the following command:

```
import pandas as pd
laws = pd.read_csv("data/laws_topic_model.csv")
```

If you are working in Colab, you can read in the file directly from GitHub using pandas:

```
import pandas as pd
laws = pd.read_csv('https://raw.githubusercontent.com/nuitrcs/AI_Week_Topic_Model/refs/heads/main/exercises/data/laws_topic_model.csv')
```

### Hotel Reviews Dataset ([source](https://data.mendeley.com/datasets/s62ycm698z/2))

If you are working on a notebook in the `exercises/` directory and you want to use the `bali_hotel_reviews.csv` dataset, you can run the following command:

```
import pandas as pd
reviews = pd.read_csv("data/bali_hotel_reviews.csv")
```

If you are working in Colab, you can read in the file directly from GitHub using pandas:

```
import pandas as pd
reviews = pd.read_csv('https://raw.githubusercontent.com/nuitrcs/AI_Week_Topic_Model/refs/heads/main/exercises/data/bali_hotel_reviews.csv')
```

### News Headlines Dataset ([source](https://huggingface.co/datasets/valurank/News_headlines))

Use the code below to pull the data from huggingface using their API.  Note that you may receive a warning in your notebook saying that the "secret `HF_TOKEN` does not exist"; this is fine.  You do not need to authenticate or create a token to access this dataset.  The data should still be pulled into your notebook and available in the df variable.

```
import pandas as pd

splits = {'train': 'final_headline_train_12000.csv', 'validation': 'final_headline_valid_1200.csv'}
df1 = pd.read_csv("hf://datasets/valurank/News_headlines/" + splits["train"])
df2 = pd.read_csv("hf://datasets/valurank/News_headlines/" + splits["validation"])
df = pd.concat([df1, df2], ignore_index=True)
```

### 20 Newsgroups ([source](https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups))

This is the same dataset that we used for our example and exercise 1. The example code below extracts all available categories, but you can also limit the categories as we did in our previous code. 

```
from sklearn.datasets import fetch_20newsgroups

bunch = fetch_20newsgroups(remove=("headers","footers","quotes"))

docs = bunch["data"]
doc_labels = bunch["target"]

df = pd.DataFrame({
    "text": docs,
    "labels": doc_labels
})

# create a label with text info
df["labels_text"] = df["labels"].astype("category").cat.rename_categories({i:j for i,j in enumerate(bunch["target_names"])})
```
