"""Read labels used in older Book files without rewriting the author’s text."""

# These are serialized field identifiers from the original file format.
FIELD_NAMES = {
    "\u7b80\u4ecb": "summary",
    "\u6982\u8ff0": "summary",
    "\u89d2\u8272\u7b80\u4ecb": "summary",
    "\u4eba\u7269\u7b80\u4ecb": "summary",
    "\u53d8\u5316": "change",
    "\u72b6\u6001\u53d8\u5316": "change",
    "\u5f27\u5149": "change",
    "\u53d1\u5c55": "change",
    "\u76ee\u6807": "goal",
    "\u52a8\u673a": "motivation",
    "\u9a71\u52a8\u529b": "motivation",
    "\u51b2\u7a81": "conflict",
    "\u538b\u529b": "conflict",
    "\u8eab\u4efd": "role",
    "\u5b9a\u4f4d": "role",
    "\u804c\u4e1a": "role",
    "\u5173\u7cfb": "relationships",
    "\u4eba\u9645": "relationships",
    "\u9650\u5236": "boundary",
    "\u8fb9\u754c": "boundary",
    "\u5e95\u7ebf": "boundary",
    "\u7981\u5fcc": "boundary",
    "\u59d3\u540d": "name",
    "\u540d\u5b57": "name",
    "\u89d2\u8272\u540d": "name",
    "\u4eba\u7269\u540d": "name",
    "\u8bbe\u5b9a\u540d\u79f0": "name",
    "\u540d\u79f0": "name",
    "\u7c7b\u522b": "category",
    "\u7c7b\u578b": "category",
    "\u89c4\u5219": "rule",
    "\u5f71\u54cd": "impact"
}

LEGACY_BRIEF_HEADER = "[NovelClaw \u751f\u6210\u7b80\u62a5]"
WORLD_NAME_MARKERS = ('\u89c4\u5219', '\u8bbe\u5b9a', '\u8fb9\u754c', '\u6846\u67b6', '\u751f\u6001')
WORLD_TEXT_MARKERS = ('\u89c4\u5219', '\u8bbe\u5b9a', '\u8fb9\u754c', '\u4e8b\u5b9e\u5361')

def field_name(raw: str) -> str:
    key = raw.strip().lower()
    return FIELD_NAMES.get(key, key)
