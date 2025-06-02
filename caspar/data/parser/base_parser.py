import re
from typing import Type, Union, Optional, Dict


class BaseParser:
    TYPE = None
    HEADER = None
    LINE = None
    VERSION = None
    SPECIAL_PARSER = {}

    @classmethod
    def make_line_regex_for_groups(cls, line: str, version_field: Optional[str] = None) -> re.Pattern:
        split_line = line.split()

        post_split = map(lambda key: f"(?P<{key.lower()}>([^\t]+))", split_line)
        intermediary = [entry for entry in post_split]

        if version_field:
            version_index = split_line.index("version")
            intermediary.insert(version_index, version_field + "\.(?P<version>\d{8})")

        joined_line = "\s".join(intermediary)
        return re.compile(f"^({joined_line})$")

    @classmethod
    def make_header_regex_for(cls, header: str) -> re.Pattern:
        return re.compile(f'^({header.upper().replace(" ", "\s+")})$')

    @classmethod
    def accepts(cls, line: str) -> bool:
        """Check if the line matches the parser's header or line pattern."""
        if cls.HEADER.match(line):
            return True
        match = cls.LINE.match(line)
        return match and match.group("version") == cls.VERSION

    @classmethod
    def is_parseable_line(cls, line: str) -> bool:
        """Check if the line is parseable by this parser."""
        match = cls.LINE.match(line)
        return match and match.group("version") == cls.VERSION

    @classmethod
    def parse(cls, line: str) -> Optional[Dict[str, Union[str, int, float]]]:
        """Parse the line and return a dictionary of parsed values."""
        if cls.is_parseable_line(line):
            return cls.parse_line(line)
        return None

    @classmethod
    def parse_line(cls, line: str) -> Dict[str, Union[str, int, float]]:
        """Parse the line and return a dictionary of parsed values."""
        match = cls.LINE.match(line)
        if not match:
            raise ValueError(f"Line does not match expected format: {line}")
        data = match.groupdict()
        parsed_data = {key: cls._convert_value(key, value) for key, value in data.items()}
        parsed_data["_type"] = cls.TYPE
        return parsed_data

    @classmethod
    def _convert_value(cls, key: str, value: str):
        """Convert the value to the appropriate type."""
        if key in cls.SPECIAL_PARSER:
            return cls.SPECIAL_PARSER[key](value)
        if value.isdigit():
            return int(value)

        try:
            return float(value)
        except ValueError:
            return value.strip()


class ParserFactory:
    """Factory class to create parsers based on the line type."""

    @staticmethod
    def get_parser(line: str) -> Optional[Type[BaseParser]]:
        """Return the appropriate parser for the given line."""
        for parser in BaseParser.__subclasses__():
            if parser.accepts(line):
                return parser
        return None
