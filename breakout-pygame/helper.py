import matplotlib.pyplot as plt
from IPython import display

plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))  

def plot(scores, mean_scores, median_scores):
    ax.clear()
    ax.set_title('Training...')  # Set title
    ax.set_xlabel('Number of Games')  # Set x-axis label
    ax.set_ylabel('Score')  # Set y-axis label
    ax.plot(scores, label='Scores')
    ax.plot(mean_scores, label='Mean Scores')
    ax.plot(median_scores, label='Median Scores')
    ax.legend()
    ax.set_ylim(bottom=0)  # Set y-axis lower limit
    ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    ax.text(len(median_scores)-1, median_scores[-1], str(median_scores[-1]))
    display.display(fig)  # Display the updated plot
    display.clear_output(wait=True)  # Clear the output
    plt.pause(0.1)