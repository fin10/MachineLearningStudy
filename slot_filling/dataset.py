import re


class Dataset:
    __iob_regex = re.compile('\(([^)]+)\)\[([\w]+)\]')
    __domains = []
    __slots = []

    @classmethod
    def init_slot(cls, slots: set):
        cls.__slots = list(slots)
        cls.__slots.sort()
        pass

    def __init__(self):
        self.__domain = []
        self.__raw = []
        self.__iob = []
        self.__tokens = []

    def append(self, domain: str, raw: str):
        if len(raw) is not 0:
            if domain not in Dataset.__domains:
                Dataset.__domains.append(domain)

            self.__domain.append(domain)
            self.__raw.append(raw)

            iob, tokens = Dataset.parse_iob(domain, raw)
            self.__iob.append(iob)
            self.__tokens.append(tokens)

    def get_domain(self, index: int = None):
        return index is None and self.__domain or self.__domain[index]

    def get_raw(self, index: int):
        return self.__raw[index]

    def get_iob(self, index: int = None):
        return index is None and self.__iob or self.__iob[index]

    def get_tokens(self, index: int = None):
        return index is None and self.__tokens or self.__tokens[index]

    def get_dataset(self, domain: str):
        dataset = Dataset()
        indexes = [index for index in range(len(self.__domain)) if self.__domain[index] == domain]
        dataset.__raw = [self.__raw[index] for index in indexes]
        dataset.__iob = [self.__iob[index] for index in indexes]
        dataset.__tokens = [self.__tokens[index] for index in indexes]

        return dataset

    def length(self):
        return len(self.__raw)

    @classmethod
    def get_domains(cls):
        return cls.__domains

    @classmethod
    def get_slots(cls):
        return cls.__slots

    @staticmethod
    def create_dataset(lines: list):
        data_set = Dataset()

        regex = re.compile('\[([^\]]+)\] (.+)')
        for line in lines:
            match = regex.findall(line)[0]
            data_set.append(match[0].strip(), match[1].strip())

        return data_set

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
                raise ValueError('{} is not invalid from \'{}\'.'.format(slot, raw))

        return iob, tokens
