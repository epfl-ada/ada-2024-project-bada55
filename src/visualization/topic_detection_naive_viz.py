import matplotlib.pyplot as plt

def fig_show_wordcloud(wordcloud, figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.close()
    return fig