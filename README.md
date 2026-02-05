# Eurovision Network Analysis 🎵🌍

A comprehensive Social Network Analysis (SNA) project exploring voting patterns and relationships between countries in the Eurovision Song Contest. This project uses network analysis techniques to uncover communities, voting trends, and geographic influences in Eurovision voting data from 1975 to 2021.

## 📊 Project Overview

This repository contains network analysis of Eurovision Song Contest voting data, examining how countries vote for each other and identifying voting patterns, communities, and potential biases. The analysis includes both jury votes and televotes from the Eurovision finals.

### Key Features

- **Network Visualization**: Interactive visualizations of voting networks with country flags
- **Community Detection**: Identification of voting communities using clustering algorithms
- **Centrality Analysis**: Multiple centrality measures (PageRank, Betweenness, Closeness, Eigenvector)
- **Geographic Analysis**: Spatial visualization of voting patterns on a map
- **Statistical Metrics**: Density analysis, degree assortativity, and network metrics
- **Regional Comparisons**: East vs. West voting pattern analysis

## 🗂️ Repository Structure

```
Eurovision-Network-Analysis/
├── datas/                              # Data directory
│   ├── Eurovision_juryvotes_2021.csv  # 2021 jury votes
│   ├── Eurovision_televotes_2021.csv  # 2021 televotes
│   ├── df_final_2017.csv              # Preprocessed 2017 data
│   ├── df_final_2018.csv              # Preprocessed 2018 data
│   ├── df_final_2019.csv              # Preprocessed 2019 data
│   ├── eurovision_song_contest_1975_2019.xlsx  # Historical data
│   ├── countries.csv                   # Country metadata (coordinates, codes)
│   ├── flags.zip                       # Country flag images
│   └── sna_eurovision_project.py      # Helper functions
├── SNA_Eurovision_Analysis_Aspromonte_Romito.ipynb  # Main analysis notebook
├── Cleaning_Eurovision.ipynb          # Data preprocessing notebook
├── Report.pdf                          # Detailed project report
└── README.md                           # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python libraries (see Installation)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sustaz/Eurovision-Network-Analysis.git
cd Eurovision-Network-Analysis
```

2. Install required dependencies:
```bash
pip install networkx pandas numpy matplotlib scipy colorthief tqdm jupyter
```

3. Extract the flags:
```bash
cd datas
unzip flags.zip
cd ..
```

### Usage

#### Running the Main Analysis

1. Open the main analysis notebook:
```bash
jupyter notebook SNA_Eurovision_Analysis_Aspromonte_Romito.ipynb
```

2. Run all cells to:
   - Load and preprocess Eurovision voting data
   - Build directed weighted networks
   - Compute network metrics and centrality measures
   - Generate visualizations and community detection
   - Create the final Eurovision voting map

**Note**: The notebook requires all files in the `datas` folder **except** the `.xlsx` file (which is only needed for data preprocessing).

#### Data Preprocessing (Optional)

To generate new dataframes or extract data for years before 2017:

1. Open the preprocessing notebook:
```bash
jupyter notebook Cleaning_Eurovision.ipynb
```

2. This notebook:
   - Reads the `eurovision_song_contest_1975_2019.xlsx` file
   - Cleans and preprocesses the data
   - Generates yearly CSV files (e.g., `df_final_2017.csv`)

**Note**: Preprocessed dataframes for 2017-2019 are already included in the `datas` folder.

## 📈 Analysis Highlights

### Network Metrics
- **Degree Centrality**: Identifies countries that receive/give the most votes
- **Betweenness Centrality**: Finds countries that act as bridges in voting patterns
- **PageRank**: Determines the most "important" countries in the voting network
- **HITS Algorithm**: Identifies hubs (countries that give many votes) and authorities (countries that receive many votes)

### Community Detection
- **Hierarchical Clustering**: Groups countries based on similar voting patterns
- **Girvan-Newman Algorithm**: Detects natural communities in the voting network
- **Geographic Analysis**: Compares East vs. West voting trends

### Visualizations
- Network graphs with geographic positioning
- Dendrogram of country similarities
- Interactive network with country flags
- Centrality heatmaps

## 📊 Data Sources

- **Eurovision 2021 Data**: Jury and televote data from the 2021 Eurovision Song Contest
- **Historical Data**: Eurovision voting data from 1975 to 2019
- **Country Metadata**: Geographic coordinates and country codes for visualization

## 🤝 Authors

- Aspromonte
- Romito

## 📄 License

This project is available for educational and research purposes.

## 📚 References

For detailed methodology and findings, please refer to the `Report.pdf` included in this repository.

---

**Enjoy exploring the Eurovision voting network! 🎉**
