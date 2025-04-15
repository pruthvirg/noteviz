"""
Test configuration for noteviz.
"""
from pathlib import Path

# Test mode flag
TEST_MODE = True

# Path to test PDF file
TEST_PDF_PATH = Path("tests/data/test.pdf")

# Mock responses for testing
TEST_RESPONSES = {
    "summary": "This is a test summary of the document.",
    "topics": [
        "Introduction to AI",
        "Machine Learning Basics",
        "Neural Networks",
        "Deep Learning Applications"
    ],
    "topic_summaries": {
        "Introduction to AI": "Overview of artificial intelligence and its history.",
        "Machine Learning Basics": "Fundamental concepts of machine learning.",
        "Neural Networks": "Structure and function of neural networks.",
        "Deep Learning Applications": "Real-world applications of deep learning."
    },
    "topic_descriptions": {
        "Introduction to AI": "Learn about the fundamentals of AI, its history, and core concepts.",
        "Machine Learning Basics": "Explore the basic principles and algorithms of machine learning.",
        "Neural Networks": "Understand how artificial neural networks work and their applications.",
        "Deep Learning Applications": "Discover real-world applications of deep learning in various fields."
    }
} 