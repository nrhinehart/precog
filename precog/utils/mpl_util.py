from io import BytesIO
import numpy as np
from PIL import Image

def fig2rgb_array(fig, expand=True, dpi=96):
    """Convert an mpl to a numpy array"""
    buf_ = BytesIO()
    fig.savefig(buf_, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf_.seek(0)
    image = Image.open(buf_)

    # Increate rank because tf.summmary.image expects a 4d tensor.
    npa = np.asarray(image)[None]
    buf_.close()
    return npa
