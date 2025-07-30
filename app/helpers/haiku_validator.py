import re
from typing import List, Dict, Optional, Any
import syllapy


class HaikuValidator:
    """A class for analyzing haiku syllable patterns and counting syllables in text."""

    DEFAULT_CUSTOM_SYLLABLES = {
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
    NON_ALPHA_PATTERN = re.compile(r"[^a-z]")
    VOWEL_GROUPS_PATTERN = re.compile(r"[aeiouy]+")
    CONSONANT_LE_PATTERN = re.compile(r"[^aeiou]le$")
    DIPHTHONG_PATTERNS = [
        re.compile(r"ia(?![aeiou])"),
        re.compile(r"io(?![aeiou])"),
        re.compile(r"ua(?![aeiou])"),
    ]

    def __init__(self, custom_syllables: Optional[Dict[str, int]] = None):
        """
        Initialize the HaikuAnalyzer.

        Args:
            custom_syllables: Optional dictionary of word->syllable count overrides.
                             If None, uses default custom syllables.
        """
        self.custom_syllables = custom_syllables or self.DEFAULT_CUSTOM_SYLLABLES.copy()

    def add_custom_syllable(self, word: str, count: int) -> None:
        """Add or update a custom syllable count for a word."""
        self.custom_syllables[word.lower().strip()] = count

    def remove_custom_syllable(self, word: str) -> None:
        """Remove a custom syllable count for a word."""
        self.custom_syllables.pop(word.lower().strip(), None)

    def count_syllables(self, word: str) -> int:
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

        if word_clean in self.custom_syllables:
            return self.custom_syllables[word_clean]

        try:
            count = syllapy.count(word_clean)
            if count > 0:
                return count
        except Exception:
            pass

        return self._estimate_syllables(word_clean)

    def _estimate_syllables(self, word: str) -> int:
        """
        Estimate syllables using improved vowel-grouping heuristic.

        Args:
            word: The word to estimate syllables for (should be lowercase)

        Returns:
            Estimated number of syllables (minimum 1)
        """
        if not word:
            return 0

        word = self.NON_ALPHA_PATTERN.sub("", word)

        if not word:
            return 0

        syllables = len(self.VOWEL_GROUPS_PATTERN.findall(word))

        if word.endswith("e") and len(word) > 1:
            if not (
                word.endswith("le") and len(word) > 2 and word[-3] not in "aeiou"
            ) and not word.endswith("ue"):
                syllables -= 1

        if self.CONSONANT_LE_PATTERN.search(word):
            syllables += 1

        for pattern in self.DIPHTHONG_PATTERNS:
            syllables += len(pattern.findall(word))

        return max(syllables, 1)

    def count_line_syllables(self, line: str) -> int:
        """
        Count total syllables in a line of text.

        Args:
            line: Text line to analyze

        Returns:
            Total syllable count for the line
        """
        if not line or not isinstance(line, str):
            return 0

        words = self.WORD_PATTERN.findall(line)
        return sum(self.count_syllables(word) for word in words)

    def is_haiku(self, text: str) -> bool:
        """
        Check if text follows traditional haiku syllable pattern (5-7-5).

        Args:
            text: Multi-line text to check

        Returns:
            True if text follows 5-7-5 syllable pattern, False otherwise
        """
        if not text or not isinstance(text, str):
            return False

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        if len(lines) != 3:
            return False

        syllable_counts = [self.count_line_syllables(line) for line in lines]
        return syllable_counts == [5, 7, 5]

    def get_syllable_breakdown(self, text: str) -> List[int]:
        """
        Get syllable count for each line in the text.

        Args:
            text: Multi-line text to analyze

        Returns:
            List of syllable counts for each non-empty line
        """
        if not text or not isinstance(text, str):
            return []

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return [self.count_line_syllables(line) for line in lines]

    def analyze_haiku(self, text: str) -> Dict[str, Any]:
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
                "errors": ["Empty or invalid text"],
            }

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        syllable_counts = [self.count_line_syllables(line) for line in lines]
        expected = [5, 7, 5]

        errors = []
        if len(lines) != 3:
            errors.append(f"Expected 3 lines, found {len(lines)}")

        if len(lines) == 3:
            for i, (actual, expected_count) in enumerate(
                zip(syllable_counts, expected)
            ):
                if actual != expected_count:
                    errors.append(
                        f"Line {i+1}: expected {expected_count} syllables, found {actual}"
                    )

        return {
            "is_haiku": len(errors) == 0,
            "syllable_counts": syllable_counts,
            "expected": expected,
            "line_count": len(lines),
            "errors": errors,
        }
