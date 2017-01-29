import unittest

from dataset import Dataset


class DatasetTest(unittest.TestCase):
    def test_append(self):
        domain = 'test'
        raw = 'i wanna go to (Essex Street)[to_stop]'
        iob = ['o', 'o', 'o', 'o', 'b-test.to_stop', 'i-test.to_stop']
        tokens = ['i', 'wanna', 'go', 'to', 'essex', 'street']
        dataset = Dataset()
        dataset.append(domain, raw)
        self.assertEqual(dataset.get_domain(0), domain)
        self.assertEqual(dataset.get_raw(0), raw)
        self.assertEqual(dataset.get_iob(0), iob)
        self.assertListEqual(dataset.get_tokens(0), tokens)

    def test_parse_iob(self):
        test = 'i would like to go from (Columbia University)[from_stop] to (Herald Square)[to_stop]'
        expected_iob = ['o', 'o', 'o', 'o', 'o', 'o', 'b-test.from_stop', 'i-test.from_stop', 'o', 'b-test.to_stop',
                        'i-test.to_stop']
        expected_tokens = ['i', 'would', 'like', 'to', 'go', 'from', 'columbia', 'university', 'to', 'herald', 'square']
        actual_iob, actual_tokens = Dataset.parse_iob('test', test)
        self.assertEqual(actual_iob, expected_iob)
        self.assertEqual(actual_tokens, expected_tokens)

        for slot in set(expected_iob):
            self.assertIn(slot, Dataset.get_slots())


if __name__ == '__main__':
    unittest.main()
