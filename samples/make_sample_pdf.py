from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("/home/claude/fact-check-agent/sample_trap_document.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("GlobalTech Marketing One-Pager (Sample Trap Document)", styles["Title"]))
story.append(Spacer(1, 12))

body_text = """
GlobalTech is a leader in the consumer electronics space. Our latest market
research highlights several key trends shaping the industry today.

The global smartphone market shipped 1.9 billion units in 2023, making it the
strongest year in smartphone history. Apple's iPhone commands a 45% global
market share as of 2023, far ahead of any competitor.

In the social media space, Twitter reported 330 million monthly active users
in its most recent quarterly filing. Meanwhile, OpenAI released GPT-5 to the
public in January 2023, kicking off a new wave of enterprise AI adoption.

On the hardware side, Tesla posted annual revenue of $150 billion in fiscal
year 2022, and the iPhone 15 Pro ships with 12GB of RAM, the most of any
smartphone on the market.

Looking at company milestones, Amazon was founded in 1998 and remains
headquartered in Seattle, Washington. The James Webb Space Telescope was
launched in December 2021 and has been operational ever since, providing
unprecedented views of the early universe.

Finally, India's population crossed the 1.4 billion mark in 2023, overtaking
China as the world's most populous country -- a milestone widely covered by
international media.
"""

for para in body_text.strip().split("\n\n"):
    story.append(Paragraph(para.strip().replace("\n", " "), styles["Normal"]))
    story.append(Spacer(1, 10))

doc.build(story)
print("Sample trap PDF created.")
