import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.data_preprocessing import basic_cleaning

def test_basic_cleaning_removes_urls():
    tweet = "Check this out: https://example.com"
    cleaned = basic_cleaning(tweet)
    assert "http" not in cleaned and "example.com" not in cleaned

def test_basic_cleaning_removes_mentions_and_hashtags():
    tweet = "Thanks @user for the tip! #AI"
    cleaned = basic_cleaning(tweet)
    assert "@" not in cleaned and "#" not in cleaned

def test_basic_cleaning_lowercases_text():
    tweet = "This Is A TEST"
    cleaned = basic_cleaning(tweet)
    assert cleaned == cleaned.lower()

def test_basic_cleaning_removes_punctuation():
    tweet = "Wow!!! Amazing ;)"
    cleaned = basic_cleaning(tweet)
    assert "!" not in cleaned and ";" not in cleaned and ")" not in cleaned
