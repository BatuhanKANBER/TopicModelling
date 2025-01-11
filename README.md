# LDA Topic Analysis Project

This project implements a topic modeling pipeline using Latent Dirichlet Allocation (LDA) with Turkish text data. It analyzes topics in the dataset, evaluates model coherence, and visualizes topic distributions.

## Features

1. **Preprocessing and Tokenization:**
   - Comments from the CSV file are tokenized and processed.
   - Zemberek library is used for Turkish text morphology analysis.

2. **Topic Modeling:**
   - LDA model is trained on the tokenized corpus.
   - Topics are extracted with their most significant words.

3. **Coherence Evaluation:**
   - Coherence scores are calculated to evaluate the quality of topics.
   - Optimal topic count is determined using a coherence score graph.

4. **Visualization:**
   - Bar chart, line chart, and pie chart to visualize topic distributions in the dataset.

5. **Outputs:**
   - Topics and their associated words.
   - Document-topic distributions saved to CSV files for further analysis.

## Requirements

- Python 3.8+
- Required libraries:
  ```bash
  pip install pandas gensim matplotlib jpype1 wordcloud
  ```
- [Zemberek Library](https://github.com/ahmetaa/zemberek-nlp): `zemberek-full.jar` file is required.

## File Structure

```
project-directory/
|
|-- assets/
|   |-- comments/comment.csv              # Input file with comments.
|   |-- processed_comments/processed_comment.csv # Processed comments.
|
|-- results/
|   |-- lda/topicler.csv                  # Topics and their top words.
|   |-- lda/yorumlarda_topic_dagilimlari.csv # Document-topic associations.
|   |-- lda/topic_dagilimlari.csv         # Topic distributions.
|
|-- zemberek/zemberek-full.jar            # Zemberek library.
```

## Usage

1. **Data Preparation:**
   - Ensure the `comment.csv` file is present under the `assets/comments/` directory.
   - Preprocess and tokenize the comments using Zemberek.

2. **Run the Script:**
   Execute the script to train the LDA model and generate outputs:
   ```bash
   python lda_analysis.py
   ```

3. **Analyze Results:**
   - View the optimal number of topics using the coherence score graph.
   - Explore the generated CSV files to understand topic distributions.

## Key Code Sections

### 1. Import Libraries and Initialize Zemberek
```python
import pandas as pd
from jpype import JClass, getDefaultJVMPath, startJVM
from wordcloud import WordCloud
ZEMBEREK_PATH = r'zemberek/zemberek-full.jar'
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % ZEMBEREK_PATH)
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()
```

### 2. Load and Process Data
```python
df = pd.read_csv('assets/processed_comments/processed_comment.csv')
tokenized = [comment.split() for comment in df["text"].astype(str)]
dictionary = corpora.Dictionary(tokenized)
dictionary.filter_extremes(no_below=1, no_above=0.7)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
```

### 3. Train LDA Model
```python
topicCount = 8
ldamodel = gensim.models.ldamodel.LdaModel(
    corpus, num_topics=topicCount, id2word=dictionary, passes=30, alpha='auto', eta='auto'
)
```

### 4. Coherence Score Evaluation
```python
from gensim.models.coherencemodel import CoherenceModel
coherence_model_lda = CoherenceModel(
    model=ldamodel, texts=tokenized, dictionary=dictionary, coherence='c_v'
)
coherence_lda = coherence_model_lda.get_coherence()
print('LDA Coherence Score:', coherence_lda)
```

### 5. Visualize Results
- **Coherence Score Graph:**
```python
plt.plot(topicCountList, coherenceCountList, '-')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.show()
```
- **Bar Chart:**
```python
plt.bar(topicDists.index, topicDists.values, color='black', alpha=0.3)
plt.xlabel('Topic ID')
plt.ylabel('Document Count')
plt.show()
```
- **Pie Chart:**
```python
plt.pie(topicDists.values, labels=topicDists.index, autopct='%1.1f%%', startangle=90)
plt.show()
```

## Outputs

- **Topic Words:** Stored in `results/lda/topicler.csv`.
- **Document-Topic Distribution:** Stored in `results/lda/yorumlarda_topic_dagilimlari.csv` and `results/lda/topic_dagilimlari.csv`.

## Example Visualizations

1. **Coherence Score Graph:**
   ![Coherence Score Graph](example_coherence_score.png)

2. **Topic Distribution Pie Chart:**
   ![Topic Pie Chart](example_pie_chart.png)

## License

This project is open-source and licensed under the MIT License.
