# src/constants/emotions.py

EMOTION_LABELS = {
    i: label for i, label in enumerate([
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
        'remorse', 'sadness', 'surprise', 'neutral'
    ])
}

# EMOTION_LABELS = [
#     'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
#     'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
#     'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
#     'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
#     'remorse', 'sadness', 'surprise', 'neutral'
# ]
# appel :
# from constants.emotions import EMOTION_LABELS as emotion_cols