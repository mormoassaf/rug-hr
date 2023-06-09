import numpy as np
import matplotlib.pyplot as plt


def show_labelled_manuscript(manuscript,
                             params=None,
                             figsize=(15, 15),
                             random_crop_size=(256, 256),
                             nlabels=128, interactive=True):
    if random_crop_size:
        i = np.random.randint(0, manuscript.shape[0] - random_crop_size[0])
        j = np.random.randint(0, manuscript.shape[1] - random_crop_size[1])
        manuscript = manuscript[i:i + random_crop_size[0], j:j + random_crop_size[1]]
    if params:
        formatted_string = ", ".join([f"{k}: {v}" for k, v in params.items()])
    if not interactive:
        plt.figure(figsize=figsize)
        plt.imshow(nlabels - manuscript, cmap="CMRmap")
        if params:
            plt.title(formatted_string)
        plt.axis('off')
        plt.show()
    else:
        # explore output interactively using plotly
        import plotly

        # plot output matrix which is (w, h) shaped and allow user to zoom in and explore
        # plot with cmap that gives a good contrast between each intensity value and color map that gives each color a unique color and background is white
        plotly.offline.init_notebook_mode(connected=True)
        plotly.offline.iplot({
            "data": [{
                "z": np.flip(manuscript, axis=0),
                "type": "heatmap",
                "colorscale": "Greys",
            }],
            "layout": {
                "title": "Manuscript" if not params else formatted_string,
                "xaxis": {
                    "title": "Width",
                },
                "yaxis": {
                    "title": "Height",
                },
            }
        })

    print(manuscript.shape, len(np.unique(manuscript) - 1), "labels are present")
