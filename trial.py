from transformers import pipeline

# Load pre-trained summarization pipeline (BART or T5)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example long Wikipedia movie plot (1000+ words)
movie_plot = """
In a small town in the western United States, a mysterious stranger arrives at a remote motel where a series of events 
unfold. The stranger, who has no memory of his past, soon finds himself in a deadly struggle for survival when he is 
caught in a criminal conspiracy involving a corrupt sheriff, a local gang, and a beautiful woman who may or may not be 
on his side. As the plot thickens, the stranger uncovers a dark secret about his own past that will change everything he 
thought he knew about his life. The small town setting becomes a battleground, with everyone trying to outwit the others 
for control of a vast fortune hidden in the area. With each passing moment, the stranger finds himself more deeply 
involved in the conspiracy, and soon it becomes clear that the only way out is to take down the people who are trying 
to kill him. In the end, the stranger must decide who to trust and whether the money is worth the cost of his life. 
The plot is filled with twists, turns, and intense action, leading to a dramatic conclusion that will leave audiences 
on the edge of their seats.
"""

# Get the summary
summary = summarizer(movie_plot, max_length=150, min_length=50, do_sample=False)

# Print the generated summary
print("Generated Summary:", summary[0]['summary_text'])
