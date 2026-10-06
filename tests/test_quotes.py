import pytest

from llm.agent import quote_in_chunk

YUAN = (
    "The Yuan dynasty (Chinese: 元朝), officially the Great Yuan, was the empire "
    "or ruling dynasty of China established by Kublai Khan, leader of the "
    "Mongolian Borjigin clan. Although the Mongols had ruled territories "
    "including today's North China for decades, it was not until 1271 that "
    "Kublai Khan officially proclaimed the dynasty in the traditional Chinese "
    "style."
)
OXYGEN = (
    'Priestley published his findings in 1775 in a paper titled "An Account of '
    'Further Discoveries in Air" which was included in the second volume of his '
    "book. Because he published his findings first, Priestley is usually given "
    "priority in the discovery."
)


@pytest.mark.parametrize(
    "quote, chunk",
    [
        ("established by Kublai Khan", YUAN),
        # Real quotes from a run, stitching sentences with an ellipsis.
        (
            "ruling dynasty of China established by Kublai Khan... it was not "
            "until 1271 that Kublai Khan officially proclaimed the dynasty",
            YUAN,
        ),
        (
            "Priestley published his findings in 1775 in a paper titled 'An "
            "Account of Further Discoveries in Air'... Because he published his "
            "findings first, Priestley is usually given priority in the discovery.",
            OXYGEN,
        ),
        ("established by Kublai Khan … officially proclaimed the dynasty", YUAN),
        ("established by Kublai Khan [...] officially proclaimed the dynasty", YUAN),
        ("...officially proclaimed the dynasty", YUAN),
        ("he said... and left", "Then he said... and left."),
    ],
)
def test_quote_found(quote, chunk):
    assert quote_in_chunk(quote, chunk)


@pytest.mark.parametrize(
    "quote",
    [
        "",
        "...",
        "established by Genghis Khan",
        # Each part is real, but in the wrong order.
        "officially proclaimed the dynasty ... established by Kublai Khan",
        # A part with an invented word.
        "established by Kublai Khan ... officially proclaimed the empire",
        # Parts too short to mean anything on their own.
        "The ... Kublai ... 1271",
    ],
)
def test_quote_rejected(quote):
    assert not quote_in_chunk(quote, YUAN)
