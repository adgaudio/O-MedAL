#O-MedAL: Online Active Deep Learning for Medical Image Analysis

Code for the paper, written with pytorch.


## Setup

- Git clone the repo, and navigate your shell to the root of the repo.

- Download the Messidor dataset to ./data/messidor/

    $ ls data/messidor
    Annotation_Base11.csv  Annotation_Base14.csv  Annotation_Base23.csv  Annotation_Base32.csv  Base11  Base14  Base23  Base32
    Annotation_Base12.csv  Annotation_Base21.csv  Annotation_Base24.csv  Annotation_Base33.csv  Base12  Base21  Base24  Base33
    Annotation_Base13.csv  Annotation_Base22.csv  Annotation_Base31.csv  Annotation_Base34.csv  Base13  Base22  Base31  Base34

- Link your ~/.torch to ./data/torch (avoid downloading pre-trained if you don't need to)

    $ ln -sr ~/.torch ./data/torch

- Install missing python requirements (if necessary)

    $ cat ./requirements.txt

- You should have a gpu on the machine too!  Check with:

    $ nvidia-smi


## Usage

From the root of this repo, type:

    python -m medal OnlineMedalResnet18BinaryClassifier --run-id test -h
    python -m medal -h

## The code structure:

  - `medal/model_configs/medal.py` - **the primary source code of
    interest for this paper.**

  - `medal/model_configs` - contains the collection of model, loss
    function, default hyperparameters, data loader, etc.  Note that any
    class variables defined here are magically exposed to the
    commandline, which is very handy for quick experimentation :)

  - `medal/datasets.py` - the pytorch DataSet for Messidor

  - `medal/models` - contains some simple pytorch model files

## Reproducibility

The experiments over values of p used to produce the final online
medal results is `./bin/reproduce_paper_results.sh`.  You may need to
just run the "python -m ..." bit.  Sorry if this is confusing.

Also keep in mind if reproducing that we fixed Messidor errata for the
online portion of these results, as mentioned in the paper.

I can create a separate GitHub repo to share about 100mb of precisely
detailed log files and post-analysis data.  If there is interest for
this, please open an issue.

I created a git tag to synchronize the code with published version on
arXiv.

## Questions?

Please feel free to open an issue or send me an email.
