# Exercise 2: Applying Topic Modeling to New Text Data

In this exercise, you'll apply the topic modeling workflow to a new dataset of your choice. The goal is to deepen your understanding and get hands-on experience on the process. We encourage you to explore different models and parameter settings to find interesting patterns in the text data.



## Step 1: Choose Your Dataset

You have a few options for selecting your text data:

- Preloaded datasets: Several datasets are prepared and available for you in the `data/` folder in this repo. If you decide to use one of these datasets, you can read in the dataset using the `pandas` Python package. We also found some datasets on huggingface that you can easily retrieve to your system via the huggingface API. Examples for reading in the preloaded datasets is provided at the end of this document.

- Custom dataset: You are also welcome to find and use your own text dataset.

**Tip:** Try to choose a dataset with enough textual content (at least several hundred documents)



## Step 2: Review and Apply the Topic Modeling Workflow

Use the slides, demo code, and insights from Exercise 1 to guide your implementation. Your main steps include:

1. **Embedding**: Use a SentenceTransformer model to convert your documents into vector representations. Try comparing models like `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, etc.  See the BERTopic documentation [here](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html).
2. **Dimension reduction**: Apply `UMAP` or other tools (e.g., `PCA`, `t-SNE`) to reduce the embedding dimensionality. Tune parameters like to get optimal results.  See the BERTopic documentation [here](https://maartengr.github.io/BERTopic/getting_started/dim_reduction/dim_reduction.html).
3. **Clustering**: Cluster the reduced embeddings using `HDBSCAN` or other clustering algorithms (e.g., `k-Means`) and change parameters as needed to retrieve the best clustering. See the BERTopic documentation [here](https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html).
4. **Representation**: Generate topic representations using BERTopic or other methods. Adjust parameters like the number of top words or the vectorizer settings.  See the BERTopic documentation [here](https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html#improving-default-representation) and [here](https://maartengr.github.io/BERTopic/getting_started/representation/representation.html).



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

- Preprocessing is often an important step. Explore the raw text data first and check to see if there are texts that are empty, full of blanks, or too short that it doesn't have any meaningful content.  If so, you can write code (like in our `supplementary_code.ipynb` notebook) to remove rows of your dataframe that don't have meaningful content.
- Topic modeling is often exploratory — it's okay to not find perfect topics on the first try!
- Use visual tools (like `.visualize_topics()` in BERTopic) to understand your results.

**Good luck! And don’t hesitate to ask questions or collaborate with others as you go.**



## Datasets

I provide code below to get you started reading in the data and identifying the "docs" that you would pass to BERTopic.  The code below assumes that you are working on a notebook in the `exercises/` directory on your local computer.  If you are working on Colab, you can either upload the data directory using the sidebar, or you can read directory from GitHub by prepending the following url : `https://raw.githubusercontent.com/nuitrcs/AI_Week_Topic_Modeling/refs/heads/main/exercises/` to the `pd.read_csv` command provided in the subsections below, e.g., for the law dataset you would use the code 

```

df = pd.read_csv("https://raw.githubusercontent.com/nuitrcs/AI_Week_Topic_Modeling/refs/heads/main/exercises/data/us_federal_laws.csv")

```


### Laws Dataset ([source](https://enjalot.github.io/latent-scope/us-federal-laws))

**Example Research Question:** What are the most common topics in US Federal Laws?  How does the proportion of laws for each of these topics change with time?

```
import pandas as pd
laws = pd.read_csv("data/us_federal_laws.csv")
docs = laws["Title"].to_list()
```


### Hotel Reviews Dataset ([source](https://data.mendeley.com/datasets/s62ycm698z/2))

**Example Research Question:** What are the most common themes that guests include in their hotel reviews?  What are the most common negative reviews?  (Can we make recommendations to the company for improvements based on these results?)  Can we group these topics together hierarchically?

```
import pandas as pd
reviews = pd.read_csv("data/bali_hotel_reviews.csv")
docs = reviews["Review"].to_list()
```

### UCI Product Classification Dataset ([source](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering))

**Example Research Question:** Can we use product titles, as might be common on e-commerce websites, to group similar products together in order to provide an end user with comparisons of similar products (e.g., to provide recommendations and/or to return quality search results for their queries)?  Can we group these products together hierarchically?

```
import pandas as pd
df = pd.read_csv("data/pricerunner_aggregate.csv")
docs = df["Product Title"].to_list()
```

### 20 Newsgroups ([source](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset))

This is the same dataset that we used for our example and exercise 1. The example code below extracts all available categories, but you can also limit the categories as we did in our previous code (e.g., see the `supplementary_code.ipynb` file in this repo). 

**Example Research Question:** What topics are people discussing currently on the internet in forums/newgroups?  How do these topics relate to each other hierarchically?

```
from sklearn.datasets import fetch_20newsgroups
bunch = fetch_20newsgroups(remove=("headers","footers","quotes"))
docs = bunch["data"]
```
