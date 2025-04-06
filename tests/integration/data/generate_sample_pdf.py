"""Script to generate a sample PDF file for testing."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_sample_pdf(output_path: Path):
    """Create a sample PDF file with AI/ML content."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = 72
    y = height - margin  # Starting y position
    line_height = 15
    section_spacing = 20
    page_bottom = margin

    def add_text(text: str, font_name: str = "Helvetica", font_size: int = 10, is_bold: bool = False):
        """Add text with proper wrapping and page breaks."""
        nonlocal y
        if is_bold:
            c.setFont(f"{font_name}-Bold", font_size)
        else:
            c.setFont(font_name, font_size)
        
        for line in _wrap_text(text, 70):
            if y < page_bottom + line_height:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height

    def add_section(title: str, content: list[str]):
        """Add a section with title and content."""
        nonlocal y
        if y < page_bottom + 2 * line_height:
            c.showPage()
            y = height - margin
        
        # Add section title
        add_text(title, "Helvetica", 12, True)
        y -= line_height
        
        # Add section content
        for item in content:
            if y < page_bottom + line_height:
                c.showPage()
                y = height - margin
            add_text(item)
            y -= line_height

    # Title
    add_text("Understanding Neural Networks and Deep Learning", "Helvetica", 16, True)
    y -= section_spacing

    # Introduction
    intro_text = (
        "Neural networks are the foundation of modern deep learning systems. They are inspired by "
        "the biological neural networks in human brains. A neural network consists of layers of "
        "interconnected nodes, each processing and transmitting information to the next layer."
    )
    add_text("Introduction to Neural Networks", "Helvetica", 12, True)
    y -= line_height
    add_text(intro_text)
    y -= section_spacing

    # Basic Components
    components = [
        "1. Neurons: Basic processing units that receive inputs and produce outputs",
        "2. Weights: Parameters that determine the strength of connections",
        "3. Bias: Additional parameters that help adjust the output",
        "4. Activation Functions: Non-linear functions that introduce complexity"
    ]
    add_section("Basic Components of Neural Networks", components)

    # Network Architecture
    architectures = [
        "1. Feedforward Networks: Information flows in one direction",
        "2. Convolutional Networks: Specialized for processing grid-like data",
        "3. Recurrent Networks: Can process sequential data",
        "4. Transformer Networks: Use attention mechanisms for parallel processing"
    ]
    add_section("Network Architecture", architectures)

    # Training Process
    training_steps = [
        "1. Forward Propagation: Data flows through the network",
        "2. Loss Calculation: Error is measured against expected output",
        "3. Backpropagation: Error is used to update weights",
        "4. Optimization: Weights are adjusted using gradient descent"
    ]
    add_section("Training Process", training_steps)

    # Applications
    applications = [
        "1. Computer Vision: Image recognition and object detection",
        "2. Natural Language Processing: Text analysis and translation",
        "3. Speech Recognition: Converting speech to text",
        "4. Recommendation Systems: Personalized content suggestions"
    ]
    add_section("Applications of Neural Networks", applications)

    # Challenges and Solutions
    challenges = [
        "1. Overfitting: Model performs well on training data but poorly on new data",
        "2. Vanishing Gradients: Difficulty in training deep networks",
        "3. Computational Requirements: Need for significant processing power",
        "4. Data Requirements: Need for large amounts of training data"
    ]
    add_section("Challenges and Solutions", challenges)

    # Future Developments
    future_trends = [
        "1. Self-supervised Learning: Learning from unlabeled data",
        "2. Neural Architecture Search: Automated network design",
        "3. Quantum Neural Networks: Integration with quantum computing",
        "4. Explainable AI: Making neural networks more interpretable"
    ]
    add_section("Future Developments", future_trends)

    # Conclusion
    conclusion_text = (
        "Neural networks continue to evolve and transform various fields. Understanding their "
        "components, architectures, and training processes is crucial for developing effective "
        "deep learning solutions. As research progresses, we can expect more efficient and "
        "capable neural network models."
    )
    add_text("Conclusion", "Helvetica", 12, True)
    y -= line_height
    add_text(conclusion_text)

    c.save()


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to fit within specified width."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


if __name__ == "__main__":
    output_path = Path(__file__).parent / "sample.pdf"
    create_sample_pdf(output_path)
    print(f"Created sample PDF at: {output_path}") 