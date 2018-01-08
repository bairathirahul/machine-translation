# Assignment 6

The solutions to the assignment 6 are written in two files; solution1.py and solution2.py

## Solution1.py
This file contains solution to the problem 1 (Direct Machine Translation). The input to this translation is stored in the "data" folder with filenames "dev_de.txt" (Dev set in Deutsche), "dev_en.txt" (English translation of the dev set), "test_de.txt", (Test set in Deutsche) and "test_en.txt" (English translation of the test set). The program also needs "dictionary.json" file, which is a closed dictionary. To execute the program, run following command:

`python3 solution1.py`

All required file names are hardcoded in the program. Make sure files are located at correct folder. The output of the program will list out the sentences with their genereated translations on the terminal screen. The program will also create files "output_dev.txt" and "output_text.txt" files with the results of the execution on dev and test set respectively. The files will be created in the "data" folder only.

### Requirements
The program solution1.py executes on python3 and requires following additional packages:
* NLTK (http://www.nltk.org/install.html)
* Numpy (https://www.scipy.org/scipylib/download.html)

## Solution2.py
This file contains solution to the problem 2 (IBM Model 1 implementation). The model is trained and tested on "es-en" corpus and given file structure is used. This program trains the model using the files in "es-en/train" folder, run the tests on files in "dev" and "test" folders. The output of the translation is recorded in the file with ".translate" extension in the same folder as the test set. To execute the program, run following command:

`python3 solution2.py`

All required file names are hardcoded in the program. Make sure files are located at correct folder. The program will execute translations with and without POS tagging. The outputs of the translation will be created in following files:
* es-en/dev/newstest2012.translated: Translation of dev set without POS tag
* es-en/dev/newstest2012_pos.translated: Translation of dev set with POS tag
* es-en/test/newstest2013.translated: Translation of test set without POS tag
* es-en/test/newstest2013_pos.translated: Translation of test set with POS tag

### Requirements
The program solution1.py executes on python3 and requires following additional packages:
* NLTK (http://www.nltk.org/install.html)
* Numpy (https://www.scipy.org/scipylib/download.html)

### Translation Performance
The performace of translation is calculated using Bleu Score. The python script bleu_score.py does this job. Please note that this script requires Python 2 (not Python3) for execution. The syntax for execution is:
`python bleu_score.py <original translated file> <file with translations prepared by IBM Model>`