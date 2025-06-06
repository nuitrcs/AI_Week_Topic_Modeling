# Exercise 2: Applying Topic Modeling to New Text Data

In this exercise, you'll apply the topic modeling workflow to a new dataset of your choice. The goal is to deepen your understanding and get hands-on experience on the process. We encourage you to explore different models and parameter settings to find interesting patterns in the text data.

---

## Step 1: Choose Your Dataset

You have a few options for selecting your text data:

- Preloaded datasets: Several datasets are prepared and available for you in the `data/` folder. If you decide to use these datasets, you can read in the dataset using `pandas` python package. We are found some datasets in huggingface that you can easily retrieve to your system via huggingface API. Examples for reading in data is provided at the end of this document.

- Custom dataset: You are also welcome to find and use your own text dataset.

**Tip:** Try to choose a dataset with enough textual content (at least several hundred documents)

---

## Step 2: Review and Apply the Topic Modeling Workflow

Use the slides, demo code, and insights from Exercise 1 to guide your implementation. Your main steps include:

1. **Embedding**: Use a SentenceTransformer model to convert your documents into vector representations. Try comparing models like `all-MiniLM-L6-v2`, `paraphrase-MPNet`, etc.
2. **Dimensionality Reduction**: Apply techniques such as UMAP to reduce the embedding dimensionality. Tune parameters like `n_neighbors` or `min_dist` to get optimal results.
3. **Clustering**: Cluster the reduced embeddings using HDBSCAN or other clustering algorithms.
4. **Representation**: Generate topic representations using BERTopic or other methods. Adjust parameters like the number of top words or the vectorizer settings.

---

## Step 3: Explore and Interpret Topics

- Examine your topics. Are they coherent?
- Are there obvious or surprising themes?

---

## Step 4: Go Further – Try BERTopic Variations

Once you're comfortable with the basic setup, push further by exploring advanced features in BERTopic. For example:

- **Dynamic Topic Modeling**: Visualize how topic prevalence changes over time ([docs](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html)).
- **Hierarchical Topic Modeling**: Create a visualization that lets you see the hierarchical nature of your topics. You can use this to decide if some topics should be merged together ([docs](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html))
---

## Deliverables

At the end of the day, we would like you to share your work with others. These are examples of the materials that you can prepare to demonstrate your approach and what interesting finds you made during the exercise.

- A brief write-up or notebook summarizing your topic modeling process.
- Visualizations: topic distributions, top words per topic, or topic evolution plots.
- Notes on interesting or unexpected insights from your dataset.

---

## 💡 Tips

- Preprocessing is often an important step in analysis task. Explore the raw text data and check to see if there are texts that are empty, full of blanks, or too short that it doesn't have any meaningful content.
- Topic modeling is often exploratory — it's okay to not find perfect topics on the first try!
- Use visual tools (like `.visualize_topics()` in BERTopic) to understand your results.

---

Good luck! And don’t hesitate to ask questions or collaborate with others as you go.

## Datasets

- Laws Dataset

[source](https://enjalot.github.io/latent-scope/us-federal-laws)

If you are working on a notebook in the `exercises/` directory and you want to use the `laws_topic_model.csv` dataset, you can run the following command:

```
import pandas as pd

laws = pd.read_csv("data/laws_topic_model.csv")
```

- Hotel Reviews Dataset

[source](https://data.mendeley.com/datasets/s62ycm698z/2)

If you are working on a notebook in the `exercises/` directory and you want to use the `bali_hotel_reviews.csv` dataset, you can run the following command:

```
import pandas as pd

laws = pd.read_csv("data/bali_hotel_reviews.csv")
```

- News Headlines Dataset

[source](https://huggingface.co/datasets/valurank/News_headlines)

```
import pandas as pd

splits = {'train': 'final_headline_train_12000.csv', 'validation': 'final_headline_valid_1200.csv'}
df1 = pd.read_csv("hf://datasets/valurank/News_headlines/" + splits["train"])
df2 = pd.read_csv("hf://datasets/valurank/News_headlines/" + splits["validation"])
df = pd.concat([df1, df2], ignore_index=True)
```

- 20 Newsgroups

[source](https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups)

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
