# TIMER
## About TIMER

TIMER is a deep-learning framework for general and species-specific bacterial promoter prediction.  

The benchmark datasets can be found in `./train_data/`. The independent test sets can be found in `./test_data/`. The TIMER models is available in `./models/`. See our paper for more details.

### Requirements:
- python 3.6
- Keras==2.3.0
- numpy==1.16.0
- scikit-learn==0.20.0
- scipy==1.4.1
- tensorflow==2.3.0

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

# Example
```
# Prediction for B.amyloliquefaciens promoters:
  python TIMER.py --input examples.txt --output output.txt 
                  --seq_type fixed_length --species B.amyloliquefaciens
```
