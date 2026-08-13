# social-network-analysis

An interactive Python application for analysing and visualising social networks using graph algorithms.

## Overview

This project was developed as my final-year Computer Science project. It provides an interactive environment for exploring graph metrics, comparing manual algorithm implementations with NetworkX, visualising network structures, and experimenting with graph modifications in real time.

The application is built using Streamlit and allows users to analyse multiple datasets through an intuitive graphical interface.

---

## Features

- Interactive network visualisation
- Manual implementation of:
  - Degree Centrality
  - PageRank
  - Betweenness Centrality
  - Closeness Centrality
- Comparison with NetworkX implementations
- Runtime performance evaluation
- Interactive graph editor
- Undo and Redo functionality
- Community detection using the Girvan–Newman algorithm
- Multiple graph datasets
- Runtime scaling experiments

---

## Technologies

- Python
- Streamlit
- NetworkX
- Matplotlib
- Pandas

---

## Installation

Clone the repository

```bash
git clone https://github.com/Moawiah21/social-network-analysis-tool.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Project Structure

```
app.py
degree_centrality.py
pagerank.py
betweenness_centrality.py
closeness_centrality.py
community_detection.py
data_loader.py
evaluation.py
utils.py
requirements.txt
```

---

## Main Functionality

The application allows users to:

- Visualise social networks
- Compare manual implementations against NetworkX
- Measure algorithm runtime
- Compare centrality rankings
- Edit graphs interactively
- Detect graph communities
- Analyse algorithm scalability across increasing graph sizes

---

## Future Improvements

- Additional community detection algorithms
- Larger real-world datasets
- More graph metrics
- Export functionality
- Interactive filtering

---

## Author

Moawiah Khawaldeh
Computer Science Graduate
