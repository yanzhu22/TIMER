# TIMER
## About TIMER

TIMER is a deep-learning framework for general and species-specific bacterial promoter prediction.  

The benchmark datasets can be found in `./train_data/`. The independent test sets can be found in `./test_data/`. The TIMER models is available in `./models/`. The prediction code can be found in `./codes/`. See our paper for more details.

### Requirements:
- python 3.6
- Keras==2.3.0
- numpy==1.16.0
- scikit-learn==0.20.0
- scipy==1.4.1
- tensorflow==2.3.0
- pandas-1.1.3
- absl-py-0.7.1

# Usage
```
python TIMER.py --input INPUTFILE     query sequences to be predicted in fasta format. \
                          --output OUTPUTFILE   save the prediction results. \
                          --seq_type {full_length,fixed_length} \
                          --species SPECIESFILE \
                          --species indicates the specific species, currently we accept 'B.amyloliquefaciens' or'C. jejuni' or 'L.phytofermentans' or 'C. pneumoniae' \
                          or 'E. coli' or'H.pylori' or'L. interrogans' or'M. smegmatis' or'R.capsulatus' or'S. coelicolor' or 'S.oneidensis' or 'S.pyogenes' \
                          or 'S. Typhimurium' or'General.
```

