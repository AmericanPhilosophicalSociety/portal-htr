# portal-htr

Command line utility for generating hOCR files from an Islandora site. Segmentation is run via kraken and text recognition via TrOCR.

NOTE: This is alpha software and has not yet exposed all configuration options to the command line interface. At the moment it runs on the default kraken segmentation model and a version of TrOCR trained via transfer learning on 18th century handwritten English text.

This software only runs on Linux. Windows is not suppoted. For Windows users, use WSL. Furthermore, it assumes you have access to a GPU. Usage on CPU is untested.

## Installation

Clone the repo:

```
git clone git@github.com:AmericanPhilosophicalSociety/portal-htr.git
cd portal-htr
```

Install into a virtual environment with pip:

```
python3 -m venv .venv
source .venv/bin/activate
pip install . -e
```

Create an environment file and indicate your local variables by editing the file:

```
cp .env_example .env
```

If your Islandora site does not have the views needed for the script to work, you will need to create these as well.

## Usage

This script is invoked through the command ```ocr```. There are two valid ways of running it: first, you can supply Drupal nodes as arguments. Second, you can provide a file that contains a list of nodes. These options are mutually exclusive.

You can pass as many files as you would like at one time.

### Pass nodes

```
ocr 12345 23456 34567
```

### Pass a file

```
ocr -f nodes.lst
```

```nodes.lst``` should look like this:

```
12345
23456
34567
```