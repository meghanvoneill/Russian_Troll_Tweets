# This file includes any visualization and analysis that takes
#   place both before and after the data is mined.
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def make_wordcloud(words,title):
    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white").generate(words)
    plt.figure(figsize=(20,20))
    plt.imshow(cloud, interpolation="gaussian")
    plt.axis("off") 
    plt.title(title, fontsize=60)
    plt.show()


def save_data(): 
    return


