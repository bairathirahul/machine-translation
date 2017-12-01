import sys
import os
import collections
import string
from nltk import word_tokenize
from nltk import pos_tag
from operator import itemgetter


class Solution2 :
    """
    Implementation of IBM Model 1
    """
    def __init__(self, source_file, target_file, pos_tagging=False):
        """
        Initialize the class with the training data
        :param source_file: Training corpus in source language
        :param target_file: Training corpus in target language
        """
        
        # POS Tag?
        self.pos_tagging = pos_tagging
        
        # Read training dataset
        source_lines = self.read_text_file(source_file)
        target_lines = self.read_text_file(target_file)
        
        # Generate corpus in the form of list of tuples
        corpus = []
        for index, source_line in enumerate(source_lines):
            source_words = list(word_tokenize(source_line))
            target_words = list(word_tokenize(target_lines[index]))
            if self.pos_tagging:
                source_words = pos_tag(source_words)
                target_words = pos_tag(target_words)
            corpus.append((target_words, source_words))
        
        # Train the model
        self.model = self.train(corpus, 25)
        
    def translate(self, source_file):
        """
        Translate the given file using IBM Model 1
        :param source_file: Input file in source language
        """
        
        # Name of output file. The extension of output file is .translated
        if self.pos_tagging:
            output_file = os.path.splitext(source_file)[0] + '_pos.translated'
        else:
            output_file = os.path.splitext(source_file)[0] + '.translated'
            
        try:
            # Open output file for writing
            output_file = open(output_file, 'w')
        except:
            print('Cannot open file' + output_file + ' for writing', file=sys.stderr)
            sys.exit(1)
        
        source_lines = self.read_text_file(source_file)
        # Loop on source file line by line
        for source_line in source_lines:
            # Generate word tokens
            source_words = list(word_tokenize(source_line.strip()))
            # Preform POS tagging
            if self.pos_tagging:
                source_words = pos_tag(source_words)
                
            translated_words = []
            # Generate translated words
            for word in source_words:
                if self.model[word]:
                    translated_word = max(self.model[word].items(), key=itemgetter(1))[0]
                    translated_words.append(translated_word)
            
            # Remove POS tags
            if self.pos_tagging:
                translated_words = [word[0] for word in translated_words]
                
            # Convert words to sentences
            translated_sentence = self.words_to_sentence(translated_words)
            
            # Write translated sentence to the output file
            output_file.write(translated_sentence + '\n')
            
    @staticmethod
    def read_text_file(filename):
        """
        Read the text file
        :param filename: filename of the text file
        :return: list of lines of the text file
        """
        try:
            file = open(filename, 'r')
        except:
            print('Cannot read file ' + filename + '. Please check the path', file=sys.stderr)
            sys.exit(1)
        output = []
    
        for line in file:
            line = line.strip().lower()
            output.append(line)
        return output

    @staticmethod
    def words_to_sentence(words):
        """
        Convert words list to sentence by taking care of punctuations
        :param words: Words list
        :return: Sentence
        """
        return ''.join([word if word in string.punctuation else ' ' + word for word in words]).strip()
    
    @staticmethod
    def train(corpus, iterations=100) :
        """
        Train the model using corpus
        :param corpus: Corpus
        :param iterations: Number of iterations
        :return:
        """
        
        # Model vocabulary
        source_vocabulary = set()
        for (target_words, source_words) in corpus:
            source_vocabulary = source_vocabulary.union(set(source_words))
        
        # Initialize the probabilities of every arrangement by a uniform value
        default_probability = 1 / len(source_vocabulary)
        probabilities = collections.defaultdict(lambda: default_probability)
        
        # Initialize model
        model = collections.defaultdict(collections.defaultdict)
        
        for i in range(iterations):
            # Normalized total
            normalize_total = collections.defaultdict(lambda: 0.0)
            # Arrangement total
            arrangement_total = collections.defaultdict(lambda: 0.0)
            # Source total
            source_total = collections.defaultdict(lambda: 0.0)
            
            for (target_words, source_words) in corpus:
                # Calculate normalization factor
                for target_word in target_words:
                    normalize_total[target_word] = 0.0
                    for source_word in source_words:
                        normalize_total[target_word] += probabilities[(target_word, source_word)]
    
                # Calculate totals
                for target_word in target_words:
                    for source_word in source_words:
                        total = probabilities[(target_word, source_word)] / normalize_total[target_word]
                        arrangement_total[(target_word, source_word)] += total
                        source_total[source_word] += total
                
            # Calculate probability
            for (target_word, source_word) in arrangement_total.keys():
                probabilities[(target_word, source_word)] = arrangement_total[(target_word, source_word)] / source_total[source_word]
                
            # Convert model to a dictionary
            for target_word, source_word in probabilities:
                model[source_word][target_word] = probabilities[(target_word, source_word)]
                
        return model
    
    
print('--------------------------------------------')
print('| Executing translation                     |')
print('--------------------------------------------')
solution2 = Solution2('es-en/train/europarl-v7.es-en.es', 'es-en/train/europarl-v7.es-en.en')
print('| Translating det set                       |')
solution2.translate('es-en/dev/newstest2012.es')
print('| Translating test set                      |')
solution2.translate('es-en/test/newstest2013.es')
print('| Translation done                          |')
print('--------------------------------------------')

print('--------------------------------------------')
print('| Executing translation with POS tagging    |')
print('--------------------------------------------')
solution2 = Solution2('es-en/train/europarl-v7.es-en.es', 'es-en/train/europarl-v7.es-en.en', pos_tagging=True)
print('| Translating dev set                       |')
solution2.translate('es-en/dev/newstest2012.es')
print('| Translating test set                      |')
solution2.translate('es-en/test/newstest2013.es')
print('| Translation done                          |')
print('--------------------------------------------')
