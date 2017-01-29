import re


class Dataset:
    __iob_regex = re.compile('\(([^)]+)\)\[([\w]+)\]')
    __slots = []

    def __init__(self):
        self.__domain = []
        self.__raw = []
        self.__iob = []
        self.__tokens = []

    def append(self, domain: str, raw: str):
        if len(raw) is not 0:
            self.__domain.append(domain)
            self.__raw.append(raw)

            iob, tokens = Dataset.parse_iob(domain, raw)
            self.__iob.append(iob)
            self.__tokens.append(tokens)

    def get_domain(self, index: int):
        return self.__domain[index]

    def get_raw(self, index: int):
        return self.__raw[index]

    def get_iob(self, index: int = None):
        return index is None and self.__iob or self.__iob[index]

    def get_tokens(self, index: int = None):
        return index is None and self.__tokens or self.__tokens[index]

    @classmethod
    def get_slots(cls):
        return cls.__slots

    @staticmethod
    def create_dataset(items: list):
        train = Dataset()
        valid = Dataset()
        test = Dataset()

        for item in items:
            domain = item['domain']
            with open(item['train'], 'r', encoding='utf-8') as file:
                for line in file:
                    train.append(domain, line.strip())
            with open(item['valid'], 'r', encoding='utf-8') as file:
                for line in file:
                    valid.append(domain, line.strip())
            with open(item['test'], 'r', encoding='utf-8') as file:
                for line in file:
                    test.append(domain, line.strip())

        return train, valid, test

    @staticmethod
    def parse_iob(domain: str, raw: str):
        for match in Dataset.__iob_regex.finditer(raw):
            iob = ''
            chunks = match.group(1).split(' ')
            iob += ' '.join(list(chunks[i] + (i == 0 and '/b-' or '/i-') + domain + '.' + match.group(2)
                                 for i in range(len(chunks))))
            raw = raw.replace(match.group(0), iob.strip())

        iob = []
        tokens = []
        chunks = raw.split(' ')
        for chunk in chunks:
            if '/' in chunk:
                part = chunk.partition('/')
                iob.append(part[2])
                tokens.append(part[0].lower())
            else:
                iob.append('o')
                tokens.append(chunk.lower())

        for slot in iob:
            if slot not in Dataset.__slots:
                Dataset.__slots.append(slot)

        return iob, tokens

    def length(self):
        return len(self.__raw)
