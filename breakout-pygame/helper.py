import matplotlib.pyplot as plt
from IPython import display

plt.ion()

# Initialize the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

def plot(scores, mean_scores, median_scores, losses, total_reward):
    ax1.clear()
    ax1.set_title('Scores')  
    ax1.set_xlabel('Number of Games')  
    ax1.set_ylabel('Score')  
    ax1.plot(scores, label='Scores')
    ax1.plot(mean_scores, label='Mean Scores')
    ax1.plot(median_scores, label='Median Scores')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    ax1.text(len(median_scores)-1, median_scores[-1], str(median_scores[-1]))

    ax2.clear()
    ax2.set_title('Loss and Total Reward')  
    ax2.set_xlabel('Number of Steps')  
    ax2.set_ylabel('Loss/Reward')  
    ax2.plot(losses, label='Loss', color='red')
 
    ax2.plot(total_reward, label='Total Reward', color='blue')
    ax2.legend()
    ax2.set_ylim(bottom=-20)
    ax2.text(len(losses)-1, losses[-1], str(losses[-1]))
    ax2.text(len(total_reward)-1, total_reward[-1], str(total_reward[-1]))

    display.display(fig)
    display.clear_output(wait=True)
    plt.pause(0.1)
