import sys
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk import pos_tag
from nltk import FreqDist
import string
import json
import math
import numpy.random
import itertools


class Solution1:
    """
    Class that implements Direct Machine Translation, as required by the Problem 1
    of the assignment
    """
    
    def __init__(self, dictionary_file, training_file):
        """
        Initialize the class instance
        :param dictionary_file: The JSON file of closed dictionary
        :param training_file: Training target language file (English file)
        """
        # Read dictionary file
        self.dictionary = self.read_json_file(dictionary_file)
        # Read training file
        training_data = self.read_text_file(training_file)
        
        # Declare langauge model attributes
        self.unigram_words = None
        self.bigram_words = None
        self.unigram_pos_words = None
        self.bigram_pos_words = None
        self.unigram_pos = None
        self.bigram_pos = None
        
        # Prepare the language model
        self.train(training_data)
    
    @staticmethod
    def read_text_file(filename):
        """
        Read the text file
        :param filename: Filename of the text file
        :return: list of lines of the text file
        """
        try:
            file = open(filename, 'r')
        except IOError as e:
            print('Cannot read file ' + filename + '. Please check the path', file=sys.stderr)
            print('I/O error({0}): {1}'.format(e.errno, e.strerror), file=sys.stderr)
            sys.exit(1)
        output = []
        
        for line in file:
            line = line.strip().lower()
            output.append(line)
        return output
    
    @staticmethod
    def read_json_file(filename):
        """
        Read a json file
        :param filename: filename of the json file
        :return: dictionary object of json
        """
        try:
            file = open(filename, 'r')
        except IOError as e:
            print('Cannot read file ' + filename + '. Please check the path', file=sys.stderr)
            print('I/O error({0}): {1}'.format(e.errno, e.strerror), file=sys.stderr)
            sys.exit(1)
        return json.load(file)
    
    """
    # Google translation, not used
    @staticmethod
    def prepare_dictionary(lines, srclang, targetlang):
        words = []
        for line in lines:
            line = line.strip().lower()
            words = words + word_tokenize(line)
        words = map(lambda word: word.lower(), words)
        words = set(words)
        
        output = dict()
        translate_client = translate.Client()
        for word in words:
            output[word] = translate_client.translate(word, targetlang, source_language=srclang)
            output[word] = output[word]['translatedText']
        return output
    """
    
    @staticmethod
    def words_to_sentence(words):
        return ''.join([word if word in string.punctuation else ' ' + word for word in words]).strip()
    
    @staticmethod
    def fix_determiners(words):
        """
        Fix "A", "An", "The" determiners
        :param words: input words
        :return: fixed words
        """
        
        words_pos = pos_tag(words)
        # Indexes of words to remove
        indices_to_remove = []
        length = len(words_pos)
        for index, word in enumerate(words_pos):
            if word[1] == 'DT':
                # Determiner before pronouns
                if words_pos[index + 1][1] == 'PRP' or words_pos[index + 1][1] == 'PRP$':
                    indices_to_remove.append(index)
                # Replace "A" with "An"
                elif word[0] == 'a' and words_pos[index + 1][0].startswith(('a', 'e', 'i', 'o', 'u')):
                    words_pos[index] = ('an', words_pos[index][1])
            if index == length - 2:
                break
        
        # Remove words
        if len(indices_to_remove) > 0:
            for index in indices_to_remove:
                words_pos.pop(index)
        
        return [word[0] for word in words_pos]
    
    @staticmethod
    def remove_consecutive_prp(words):
        """
        Remove consecutive pronouns
        :param words: input words
        :return: fixed words
        """
        words_pos = pos_tag(words)
        indices_to_remove = []
        length = len(words_pos)
        for index, word in enumerate(words_pos):
            # Identify consecutive pronouns
            if word[1] in ('PRP', 'PRP$') and words_pos[index + 1][1] in ('PRP', 'PRP$'):
                indices_to_remove.append(index)
            if index == length - 2:
                break
        
        # Remove words
        if len(indices_to_remove) > 0:
            for index in indices_to_remove:
                words_pos.pop(index)
        
        return [word[0] for word in words_pos]
    
    @staticmethod
    def swap_verb_prp(words):
        """
        Swap reverse ordered verb and noun/pronoun
        :param words: input words
        :return: fixed words
        """
        words_pos = pos_tag(words)
        length = len(words_pos)
        for index, word in enumerate(words_pos):
            # Identify consecutive pronouns
            if word[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ') \
                    and words_pos[index + 1][1] in ('PRP', 'PRP$', 'NN', 'NNS', 'NNP', 'NNPS'):
                words_pos[index] = words_pos[index + 1]
                words_pos[index + 1] = word
            if index == length - 2:
                break
        
        return [word[0] for word in words_pos]
    
    @staticmethod
    def print_translation(translations, output_file):
        """
        Print and write the translation to the output file
        :param translations: Translations output list
        :param output_file: Output file instance
        """
        print('------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------',
              file=output_file)
        for translation in translations:
            print('\033[1m%s:\033[0m\n%s\n' % translation)
            print('%s:\n%s\n' % translation, file=output_file)
        print('------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------',
              file=output_file)
    
    def train(self, lines):
        """
        Training unigram, bigram, unigram with pos and bigram with pos models
        :param lines: Training lines
        """
        unigram_words = []
        bigram_words = []
        unigram_pos_words = []
        bigram_pos_words = []
        unigram_pos = []
        bigram_pos = []
        
        for line in lines:
            # Prepare word tokens
            words = word_tokenize(line)
            # Tag the tokens with POS
            words_pos = pos_tag(words)
            # Generate POS sequences
            pos = [word[1] for word in words_pos]
            
            # Prepare unigram lists with beginnging and end of sentences
            unigram_words = unigram_words + ['<s>'] + words + ['</s>']
            unigram_pos_words = unigram_pos_words + words_pos
            unigram_pos = unigram_pos + ['<s>'] + pos + ['</s>']
            
            # Prepare bigram lists for words, words_pos and pos
            bigram_words = bigram_words + list(
                ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            bigram_pos_words = bigram_pos_words + list(
                ngrams(words_pos, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            bigram_pos = bigram_pos + list(
                ngrams(words_pos, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
        
        # Generate frequency distribution of all lists
        self.unigram_words = FreqDist(unigram_words)
        self.bigram_words = FreqDist(bigram_words)
        self.unigram_pos_words = FreqDist(unigram_pos_words)
        self.bigram_pos_words = FreqDist(bigram_pos_words)
        self.unigram_pos = FreqDist(unigram_pos)
        self.bigram_pos = FreqDist(bigram_pos)
    
    def get_bigram_words_probability(self, words):
        """
        Calculate and returns bigram probability of the given arrangement of words
        :param words: Words list
        :return: Probability
        """
        
        probability = 0
        # Get vocabulary size
        vocabulary_size = len(self.unigram_words)
        # Generate bigrams
        bigrams = list(ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        # Calculate log probability with add-one smoothing
        for bigram in bigrams:
            probability += math.log(self.bigram_words.freq(bigram) + 1) - math.log(
                self.unigram_words.freq(bigram[1]) + vocabulary_size)
        
        return probability
    
    def get_bigram_pos_words_probability(self, words):
        """
        Calculates and returns bigram probability of the given arrangement of words with POS
        :param words: Words list POS tagged
        :return: Probability
        """
        # POS tag input words
        words = pos_tag(words)
        probability = 0
        # Get vocabulary size
        vocabulary_size = len(self.unigram_pos_words)
        # Generate bigrams
        bigrams = list(ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        # Calculate log probability with add-one smoothing
        for bigram in bigrams:
            probability += math.log(self.bigram_pos_words.freq(bigram) + 1) - math.log(
                self.unigram_pos_words.freq(bigram[1]) + vocabulary_size)
        
        return probability
    
    def get_bigram_pos_probability(self, tags):
        """
        Calculates and returns bigram probabilty of given arrangments of POS tags
        :param tags: Arrangement of POS tags
        :return: Probabilty
        """
        probability = 0
        # Get vocabulary size
        vocabulary_size = len(self.unigram_pos)
        # Generate bigrams
        bigrams = list(ngrams(tags, 2))
        # Calculate log probability with add-one smoothing
        for bigram in bigrams:
            probability += math.log(self.bigram_pos.freq(bigram) + 1) - math.log(
                self.unigram_pos.freq(bigram[1]) + vocabulary_size)
        
        return probability
    
    def get_highest_probability_permutation(self, words, method):
        """
        Implementation of argmax. Returns highest probability entry from the list
        words.
        :param words: List of list of words
        :param method: Method to calculate probability
        :return: Highest probability words arrangement
        """
        
        max_probability = -math.inf
        selected = None
        # Get permutation counts. If the sentence is big, limit to 100
        permutation_count = math.factorial(len(words)) if len(words) < 5 else 100
        for _ in range(permutation_count):
            # Generate random permutation
            permutation = numpy.random.permutation(words)
            # Get probability of the permutation
            probability = getattr(self, method)(permutation)
            # Select the permutation with higher probability
            if probability > max_probability:
                max_probability = probability
                selected = permutation
        
        return selected
    
    def get_arrangement_with_pos_model(self, words):
        """
        Returns arrangement of words with highest probability using POS ordering
        :param words: Input list of words
        :return: Arrangement of words with higest probability ordering
        """
        
        # Tag words with POS
        words_pos = [('', '<s>')] + pos_tag(words) + [('', '</s>')]
        length = len(words_pos)
        
        for index, word in enumerate(words_pos):
            # Pick 4 words window
            words_window = words_pos[index: index + 4]
            
            max_probability = -math.inf
            selected = None
            # Generate all permutations of the window
            permutations = itertools.permutations(words_window)
            for permutation in permutations:
                # Get all POS tags
                pos = [word[1] for word in permutation]
                # Get the probability
                probability = self.get_bigram_pos_probability(pos)
                # Pick the arrangement with the highest probability
                if probability > max_probability:
                    max_probability = probability
                    selected = permutation
            
            # Apply the arrangment to the original list of words
            words_pos[index] = selected[0]
            words_pos[index + 1] = selected[1]
            words_pos[index + 2] = selected[2]
            words_pos[index + 3] = selected[3]
            
            if index == length - 4:
                break
        
        # Return the list of rearranged words
        return [word[0] for word in words_pos]
    
    def translate(self, source_sentence, original_translation, output_file):
        """
        Perform translation of the given line
        :param source_sentence: Line of input file
        :param original_translation: Original translation from the test data
        :param output_file: File to write output to
        """
        
        # Get word tokens
        words = word_tokenize(source_sentence)
        translated_words = []
        # Perform direct machine translation using dictionary
        for i, word in enumerate(words):
            # Skip translating punctuations
            if word not in string.punctuation:
                translated_words.append(self.dictionary[word])
            else:
                translated_words.append(word)
        
        output = list()
        output.append(('Source Sentence', source_sentence))
        output.append(('Original Translation', original_translation))
        
        # Normal translation output
        translated_sentence = self.words_to_sentence(translated_words)
        output.append(('Direct Machine Translation', translated_sentence))
        
        # Improvement 1: Fixing determiners
        translated_words = self.fix_determiners(translated_words)
        translated_sentence = self.words_to_sentence(translated_words)
        output.append(('Fixing determiners', translated_sentence))
        
        # Improvement 2: Removing consecutive pronouns
        translated_words = self.remove_consecutive_prp(translated_words)
        translated_sentence = self.words_to_sentence(translated_words)
        output.append(('Removing consecutive pronouns', translated_sentence))
        
        # Improvement 3: Swapping reverse orders verbs and noun/pronouns
        translated_words = self.swap_verb_prp(translated_words)
        translated_sentence = self.words_to_sentence(translated_words)
        output.append(('Swapping reverse orders verbs and noun/pronouns', translated_sentence))
        
        # Improvement 4: Bigram Language Model
        translated_words = self.get_highest_probability_permutation(translated_words, 'get_bigram_words_probability')
        translated_sentence = self.words_to_sentence(translated_words)
        output.append(('Bigram Language Model', translated_sentence))
        
        # Improvement 5: Bigram POS Language Model
        translated_words = self.get_highest_probability_permutation(translated_words,
                                                                    'get_bigram_pos_words_probability')
        translated_sentence = self.words_to_sentence(translated_words)
        output.append(('Bigram POS Language Model', translated_sentence))
        
        # Improvement 6: Rearrangement of POS
        translated_words = self.get_arrangement_with_pos_model(translated_words)
        translated_sentence = self.words_to_sentence(translated_words)
        output.append(('Rearrangement of POS', translated_sentence))
        
        self.print_translation(output, output_file)
    
    def execute(self, input_file, translation_file, output_file):
        """
        Execute the tests on given input file
        :param input_file: Input file
        :param translation_file: File containing original translations
        :param output_file: File to write output to
        """
        input_lines = self.read_text_file(input_file)
        # Open output file for writing
        try:
            output_file = open(output_file, 'w')
        except IOError as e:
            print('Cannot open file' + output_file + ' for writing', file=sys.stderr)
            print('I/O error({0}): {1}'.format(e.errno, e.strerror), file=sys.stderr)
            sys.exit(1)
        
        original_translation_lines = self.read_text_file(translation_file)
        for index, line in enumerate(input_lines):
            # Translate each line
            self.translate(line, original_translation_lines[index], output_file)


solution1 = Solution1('data/dictionary.json', 'data/dev_en.txt')
# Execution on dev set
solution1.execute('data/dev_de.txt', 'data/dev_en.txt', 'data/output_dev.txt')
# Execution on test set
solution1.execute('data/test_de.txt', 'data/test_en.txt', 'data/output_test.txt')
