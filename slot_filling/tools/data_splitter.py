import re


class DataSplitter:
    TRAIN_RATIO = 0.6
    DEV_RATIO = 0.2
    TEST_RATIO = 0.2
    __slot_pattern = re.compile('\([^)]+\)\[([^\]]+)\]')

    @staticmethod
    def split(domain: str, data: list):
        slots = set('o')
        for item in data:
            matches = DataSplitter.__slot_pattern.findall(item)
            for match in matches:
                slots.add('b-{}.{}'.format(domain, match))
                slots.add('i-{}.{}'.format(domain, match))

        slots = list(slots)
        slots.sort()

        train_idx = int(len(data) * DataSplitter.TRAIN_RATIO)
        dev_idx = int(len(data) * DataSplitter.DEV_RATIO) + train_idx
        test_idx = int(len(data) * DataSplitter.TEST_RATIO) + dev_idx
        train = data[:train_idx]
        dev = data[train_idx:dev_idx]
        test = data[dev_idx:test_idx]

        return slots, train, dev, test
