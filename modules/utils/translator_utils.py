import base64
import ast
import json
import logging
import re
import jieba
import janome.tokenizer
import numpy as np
from pythainlp.tokenize import word_tokenize
from .textblock import TextBlock
import imkit as imk

logger = logging.getLogger(__name__)


MODEL_MAP = {
    "Custom": "",  
    "Deepseek-v3": "deepseek-chat", 
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Claude-4.6-Sonnet": "claude-sonnet-4-6",
    "Claude-4.5-Haiku": "claude-haiku-4-5-20251001",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-3.0-Flash": "gemini-3-flash-preview",
    "Gemini-2.5-Pro": "gemini-2.5-pro"
}

def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def get_raw_text(blk_list: list[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_txts_dict[block_key] = blk.text
    
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json

def get_raw_translation(blk_list: list[TextBlock]):
    rw_translations_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_translations_dict[block_key] = blk.translation
    
    raw_translations_json = json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)
    
    return raw_translations_json

def set_texts_from_json(blk_list: list[TextBlock], json_string: str):
    if not isinstance(json_string, str):
        logger.error("Translation response is not a string: %s", type(json_string).__name__)
        return

    match = re.search(r"\{[\s\S]*\}", json_string)
    if not match:
        logger.error("No JSON object found in translation response. Preview: %r", json_string[:1000])
        return

    # Extract JSON object from LLM output that may include extra prose.
    json_payload = match.group(0)
    translation_dict = _parse_translation_dict(json_payload, json_string)
    if translation_dict is None:
        translation_dict = _extract_block_pairs(json_payload)
        if translation_dict:
            logger.warning(
                "Recovered %s translation pairs from malformed JSON response.",
                len(translation_dict),
            )
        else:
            return

    if not isinstance(translation_dict, dict):
        logger.error("Parsed translation JSON is not an object: %s", type(translation_dict).__name__)
        return

    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        if block_key in translation_dict:
            blk.translation = translation_dict[block_key]
        else:
            logger.warning("Missing key in translation JSON: %s", block_key)


def _parse_translation_dict(json_payload: str, raw_response: str) -> dict | None:
    decode_error: json.JSONDecodeError | None = None

    attempts: list[str] = [json_payload]
    attempts.append(re.sub(r",\s*([}\]])", r"\1", json_payload))
    attempts.append(re.sub(r"([{,]\s*)(block_\d+)(\s*:)", r'\1"\2"\3', attempts[-1]))

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            decode_error = exc

    try:
        parsed = ast.literal_eval(json_payload)
        if isinstance(parsed, dict):
            return {str(key): value for key, value in parsed.items()}
    except Exception:
        pass

    if decode_error is not None:
        window_start = max(0, decode_error.pos - 120)
        window_end = min(len(json_payload), decode_error.pos + 120)
        logger.error(
            "Failed to parse translation JSON at line=%s col=%s pos=%s: %s",
            decode_error.lineno,
            decode_error.colno,
            decode_error.pos,
            decode_error.msg,
        )
        logger.debug("Translation raw response preview: %r", raw_response[:1200])
        logger.debug("Extracted JSON preview: %r", json_payload[:1200])
        logger.debug("JSON context around error: %r", json_payload[window_start:window_end])

    return None


def _extract_block_pairs(json_payload: str) -> dict[str, str]:
    pair_pattern = re.compile(
        r'["\']?(block_\d+)["\']?\s*:\s*("((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\')',
        flags=re.DOTALL,
    )
    recovered: dict[str, str] = {}

    for match in pair_pattern.finditer(json_payload):
        block_key = match.group(1)
        value = match.group(3) if match.group(3) is not None else match.group(4)
        recovered[block_key] = value

    return recovered

def set_upper_case(blk_list: list[TextBlock], upper_case: bool):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper() 
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def get_chinese_tokens(text):
    return list(jieba.cut(text, cut_all=False))

def get_japanese_tokens(text):
    tokenizer = janome.tokenizer.Tokenizer()
    return [token.surface for token in tokenizer.tokenize(text)]

def format_translations(blk_list: list[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    for blk in blk_list:
        translation = blk.translation
        trg_lng_code_lower = trg_lng_cd.lower()
        seg_result = []

        if 'zh' in trg_lng_code_lower:
            seg_result = get_chinese_tokens(translation)

        elif 'ja' in trg_lng_code_lower:
            seg_result = get_japanese_tokens(translation)

        elif 'th' in trg_lng_code_lower:
            seg_result = word_tokenize(translation)

        if seg_result:
            blk.translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
        else:
            # apply casing/formatting for this single block when no segmentation is done
            if translation is None:
                continue
            if upper_case and not translation.isupper():
                blk.translation = translation.upper()
            elif not upper_case and translation.isupper():
                blk.translation = translation.lower().capitalize()
            else:
                blk.translation = translation

def is_there_text(blk_list: list[TextBlock]) -> bool:
    return any(blk.text for blk in blk_list)
