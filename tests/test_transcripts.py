"""Check independent transcript comparison metrics without ASR downloads."""
import unittest
from experiment.check_transcripts import normalized_words, word_errors


class TranscriptMetricTests(unittest.TestCase):
    def test_word_errors_cover_substitution_deletion_and_insertion(self):
        reference = ['this', 'is', 'a', 'test']
        self.assertEqual(word_errors(reference, reference), 0)
        self.assertEqual(word_errors(reference, ['that', 'is', 'a', 'test']), 1)
        self.assertEqual(word_errors(reference, ['this', 'is', 'test']), 1)
        self.assertEqual(word_errors(reference, ['this', 'really', 'is', 'a', 'test']), 1)
        self.assertEqual(word_errors([], ['extra']), 1)
        self.assertEqual(word_errors(['test'], ['extra']*4), 4)

    def test_case_punctuation_and_curly_apostrophes_do_not_create_word_errors(self):
        self.assertEqual(normalized_words('It’s a TEST!'), normalized_words("it's a test."))


if __name__ == '__main__':
    unittest.main()
