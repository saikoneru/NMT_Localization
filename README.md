# Analyzing Challenges in Neural Machine Translation for Software Localization
## Documentation for reproducing experiments and/or scraping PO files from Github Repositories
### Overview
- Disambiguation Test Set
- Requirements
- Downloading WMT data
- Tokenizing and BPE (Pre-training WMT)
- Scraping and Parsing GIT po files
- Creating Context appended data sets ( + Tokenize, BPE, Preprocess)
- Training and Evaulating models (All)

## Disambiguation Test Set
The file 'data/test_disamb/en_de.disambiguation' contains the final human annotated test data (95 sentence pairs).

Each string contains the source, target and the link to the PO file (github repository) in the following format

'source', 'target', 'PO file'

## Requirements
Create a new conda environment by importing the `Kontextmt.yml` file
```sh
conda env create -n kontextmt --file kontextmt.yml
```
Then activate the conda environment
```sh
conda activate kontextmt
```
Clone [Fairseq](https://github.com/facebookresearch/fairseq), [Moses](https://github.com/moses-smt/mosesdecoder), [Subword-nmt](https://github.com/rsennrich/subword-nmt) and [CharacTER](https://github.com/rwth-i6/CharacTER) in the root directory of the repo. Follow the installation steps for fairseq in the repo
```sh
git clone https://github.com/facebookresearch/fairseq
git clone https://github.com/rsennrich/subword-nmt
git clone https://github.com/moses-smt/mosesdecoder
git clone https://github.com/rwth-i6/CharacTER
```
Install [src](https://github.com/sourcegraph/src-cli) -- Sourcegraph command line interface to search Github repos.
Install the following libraries from pip
```sh
pip install polib
pip install pathlib
pip install -U scikit-learn
```
## Downloading WMT Data
For pre-training, the data can be downloaded from [here](https://nlp.stanford.edu/projects/nmt/)
```sh
mkdir -p data/wmt_data
cd data/wmt_data
```
Create a text file `links.txt` with the following links in the root of the repo
```sh
https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en
https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de
https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en
https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de
```
Download the files using wget
```sh
wget -i links.txt
```
Then rename the files to valid and test sets
```sh
mv https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en valid.en
mv https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de valid.de
mv https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en test.en
mv https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de test.de
cd ../../scripts/
```
## Tokenizing and BPE (Pre-training WMT)
`Note: To experiment with different language pair or having a different file structure, you have to currently change the variables inside the script. If you want to add special tags that you already know during, you append your new words now. If they are unknown or will be determined, then you can later also append the embedding layer (code for extending the embedding layer will be uploaded soon)`

To tokenize the data, use the `tokenize_data.sh` file with source and target language as arguments. Then learn and apply byte-pair-encoding using the `learn_apply_bpe.sh` script. You can change the number of merge operations inside the same script
```sh
bash tokenize_data.sh "en" "de"
bash learn_apply_bpe.sh
```
This will create the `$ROOT/wmt_data/bpe/` folder with tokenized versions for train, dev and test data.
## Scraping and Parsing GIT po files
If everything is installed correctly, you can use the `get_gitdata.sh` script to download and parse po files from Github repositories. (If you do not want to download, you can use the downloaded files here `$ROOT/data/gitdata_parsed_de/`)
For example to download English to German po files, run the following command with the target language as argument in the `scripts/` directory
```sh
bash get_gitdata.sh "de"
```
It will download the files and parse them using the `extract_git_data.py` python script.
After the command is done (takes a while to process and parse multiple files), you should find `po_data_de/` with the repos and `gitdata_parsed_de/` with the raw files in the root directory
Inside the `gitdata_parsed_de/`, you should find the following files
- `full.en` : Parallel Sentence-level data for the English
- `full.de` : Parallel Sentence-level data for German
- `full.name` : File path for each sentence in `po_data_de` directory
- `full.cxt` : Filepath for each string in the source code
- `data.csv` : A csv file with all the above information and can be loaded using Pandas
## Creating Context appended data sets
Before we create the different context appended datasets, we first tokenize and apply bpe using the WMT codes with `tokenize_git_data.sh`.
```sh
bash tokenize_git_data.sh
```
Then, we use the `extract_git_cxtdata.sh` to append contexts, split in to train/dev/test, create dictionary and pre-process the WMT data and different context appended sets
```sh
bash extract_git_cxtdata.sh
```
In the script, it appends the following type of contexts
- None: no context appended
- Domain Repo name is appended to source sentence
- Neighbour The surrounding sentences (can be controlled by neighbour_size parameter)
- Filename The corresponding file in which the string is present in the source code
For each language pair, it will generate two folders for each language direction. This is done so that only the source is appended and the target is only the text without any contextual information.
Moreover, the valid and test set that are present in each folder are created using the unknown apps. Inside it, you will find the `rnd/` folder which contains the valid and test sets for known apps.
`TODO: Add document level version of data sets to implement Doc2Doc methods`
##  Training and Evaulating models
### WMT
We can train the model using `pretrain_wmt.sh` script and change source and target language direction as you need.
```sh
export CUDA_VISIBLE_DEVICES=0;export CUDA_DEVICE_ORDER=PCI_BUS_ID
bash pretrain_wmt.sh
```
After training, you can use the `eval_wmt_model.sh` to calculate the sacrebleu and characTER scores. The default variables will take the test set and evaluate. To do inference on custom data, change the variables accordingly.
### GIT
To train the context-appended data, use the `train_git_cxtdata.sh` script. This script is a loop for all training configurations and seeds. Change the `seeds` and `cxt_types` variables as needed to train only few models
```sh
bash train_git_cxtdata.sh
```
Finally for evaluation, use the `eval_git_models.sh` script. This is similar to the training script and will loop around all the different configurations.
```sh
bash eval_git_models.sh
```
