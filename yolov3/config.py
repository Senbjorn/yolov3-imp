def anchors_parser(value):
    anchors = list(map(int, value.split(',')))
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    return anchors


def list_parser(value, parser=int, separator=','):
    result = list(map(parser, value.split(separator)))
    return result


def list_int_parser(value):
    return list_parser(value)


def list_float_parser(value):
    return list_parser(value, parser=float)


PARSERS={
    'default_key': int,
    'load_skip': int,
    'activation': str,
    'policy': str,
    'jitter': float,
    'saturation': float,
    'exposure': float,
    'hue': float,
    'momentum': float,
    'decay': float,
    'learning_rate': float,
    'ignore_thresh': float,
    'truth_thresh': float,
    'angle': float,
    'anchors': anchors_parser,
    'scales': list_float_parser,
    'steps': list_int_parser,
    'mask': list_int_parser,
    'layers': list_int_parser,
}


class ValueParser:
    def parse(self, key, value):
        if key not in PARSERS:
            key = 'default_key'
        return PARSERS[key](value)


class ConfigParser:
    def _line_filter(self, line):
        if len(line) == 0:
            return False
        if line[0] == '#':
            return False
        return True
    
    def _line_preprocess(self, line):
        return line.strip()

    def parse(self, cfg_path):
        value_parser = ValueParser()
        with open(cfg_path, 'r') as config_file:
            lines = list(config_file)
        lines = list(map(self._line_preprocess, lines))
        lines = list(filter(self._line_filter, lines))
        blocks = []
        block = None
        for line in lines:
            if line[0] == '[':
                if block is not None:
                    blocks.append(block)
                block = {'type': line[1:-1].strip()}
            else:
                key, value = line.split("=")
                key, value = key.strip(), value.strip()
                block[key] = value_parser.parse(key, value)
        if block is not None:
            blocks.append(block)
        return blocks
