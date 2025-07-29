import re
from typing import List, Dict
import syllapy


CUSTOM_SYLLABLES: Dict[str, int] = {
    "haiku": 2,
    "poem": 2,
    "fire": 1,
    "every": 2,
    "world": 1,
    "orange": 2,
    "queue": 1,
    "hour": 1,
    "our": 1,
    "flower": 2,
    "power": 2,
    "tower": 2,
    "shower": 2,
    "prayer": 1,
    "player": 2,
    "layer": 2,
    "mayor": 2,
    "says": 1,
    "said": 1,
    "bread": 1,
    "dead": 1,
    "head": 1,
    "read": 1,
    "thread": 1,
    "spread": 1,
    "real": 2,
    "idea": 3,
    "create": 2,
    "ocean": 2,
    "area": 3,
    "being": 2,
    "going": 2,
    "doing": 2,
    "seeing": 2,
    "flying": 2,
    "trying": 2,
}

WORD_PATTERN = re.compile(r"[a-zA-Z'-]+")
NON_ALPHA_PATTERN = re.compile(r'[^a-z]')
VOWEL_GROUPS_PATTERN = re.compile(r'[aeiouy]+')
CONSONANT_LE_PATTERN = re.compile(r'[^aeiou]le$')
DIPHTHONG_PATTERNS = [
    re.compile(r'ia(?![aeiou])'),
    re.compile(r'io(?![aeiou])'),
    re.compile(r'ua(?![aeiou])'),
]


def count_syllables(word: str) -> int:
    """
    Return syllable count for a word using custom dictionary, syllapy library, 
    or heuristic fallback.
    
    Args:
        word: The word to count syllables for
        
    Returns:
        Number of syllables in the word (minimum 1)
    """
    if not word or not isinstance(word, str):
        return 0
        
    word_clean = word.lower().strip()
    
    if word_clean in CUSTOM_SYLLABLES:
        return CUSTOM_SYLLABLES[word_clean]

    try:
        count = syllapy.count(word_clean)
        if count > 0:
            return count
    except Exception:
        pass

    return estimate_syllables(word_clean)


def estimate_syllables(word: str) -> int:
    """
    Estimate syllables using improved vowel-grouping heuristic.
    
    Args:
        word: The word to estimate syllables for (should be lowercase)
        
    Returns:
        Estimated number of syllables (minimum 1)
    """
    if not word:
        return 0
        
    word = NON_ALPHA_PATTERN.sub('', word)
    
    if not word:
        return 0

    syllables = len(VOWEL_GROUPS_PATTERN.findall(word))

    if word.endswith('e') and len(word) > 1:
        if not (word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiou') \
           and not word.endswith('ue'):
            syllables -= 1

    if CONSONANT_LE_PATTERN.search(word):
        syllables += 1

    for pattern in DIPHTHONG_PATTERNS:
        syllables += len(pattern.findall(word))

    return max(syllables, 1)


def count_line_syllables(line: str) -> int:
    """
    Count total syllables in a line of text.
    
    Args:
        line: Text line to analyze
        
    Returns:
        Total syllable count for the line
    """
    if not line or not isinstance(line, str):
        return 0
        
    words = WORD_PATTERN.findall(line)
    return sum(count_syllables(word) for word in words)


def is_haiku(text: str) -> bool:
    """
    Check if text follows traditional haiku syllable pattern (5-7-5).
    
    Args:
        text: Multi-line text to check
        
    Returns:
        True if text follows 5-7-5 syllable pattern, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
        
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    if len(lines) != 3:
        return False

    syllable_counts = [count_line_syllables(line) for line in lines]
    return syllable_counts == [5, 7, 5]


def haiku_syllable_breakdown(text: str) -> List[int]:
    """
    Get syllable count for each line in the text.
    
    Args:
        text: Multi-line text to analyze
        
    Returns:
        List of syllable counts for each non-empty line
    """
    if not text or not isinstance(text, str):
        return []
        
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    return [count_line_syllables(line) for line in lines]


def analyze_haiku(text: str) -> Dict[str, any]:
    """
    Comprehensive haiku analysis with detailed feedback.
    
    Args:
        text: Text to analyze as potential haiku
        
    Returns:
        Dictionary containing analysis results
    """
    if not text or not isinstance(text, str):
        return {
            "is_haiku": False,
            "syllable_counts": [],
            "expected": [5, 7, 5],
            "errors": ["Empty or invalid text"]
        }
    
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    syllable_counts = [count_line_syllables(line) for line in lines]
    expected = [5, 7, 5]
    
    errors = []
    if len(lines) != 3:
        errors.append(f"Expected 3 lines, found {len(lines)}")
    
    if len(lines) == 3:
        for i, (actual, expected_count) in enumerate(zip(syllable_counts, expected)):
            if actual != expected_count:
                errors.append(f"Line {i+1}: expected {expected_count} syllables, found {actual}")
    
    return {
        "is_haiku": len(errors) == 0,
        "syllable_counts": syllable_counts,
        "expected": expected,
        "line_count": len(lines),
        "errors": errors
    }