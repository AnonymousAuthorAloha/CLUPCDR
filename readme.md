# Contrastive Learning for Extracting Transferable User Profiles in Cross-Domain Recommendation System (CLUPCDR)




## Requirements

- Python 3.6
- Pytorch > 1.0
- tensorflow
- Pandas
- Numpy
- Tqdm

## File Structure

```
.
├── code
│   ├── config.json         # Configurations
│   ├── entry.py            # Entry function
│   ├── my_model.py           # Models based on MF, GMF or Youtube DNN
│   ├── my_preprocessing.py    # Parsing and Segmentation
│   ├── readme.md
│   └── my_run.py              # Training and Evaluating 
└── data
    ├── mid                 # Mid data
    │   ├── Books.csv
    │   ├── CDs_and_Vinyl.csv
    │   └── Movies_and_TV.csv
    ├── raw                 # Raw data
    │   ├── reviews_Books_5.json.gz
    │   ├── reviews_CDs_and_Vinyl_5.json.gz
    │   └── reviews_Movies_and_TV_5.json.gz
    └── ready               # Ready to use
        ├── _2_8
        ├── _5_5
        └── _8_2
```

## Dataset

We utilized the Amazon Reviews 5-score dataset. 
To download the Amazon dataset, you can use the following link: [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/links.html).
Download the three domains: [CDs and Vinyl](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz), [Movies and TV](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz), [Books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz) (5-scores), and then put the data in `./data/raw`.

You can use the following command to preprocess the dataset. 
The two-phase data preprocessing includes parsing the raw data and segmenting the mid data. 
The final data will be under `./data/ready`.
