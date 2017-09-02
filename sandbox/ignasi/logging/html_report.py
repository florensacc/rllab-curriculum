import dominate
from dominate.tags import *
from pdb import set_trace as st

# Note here it is important to use the scipy imsave as it allows us to save
# the image into a file object
from scipy.misc import imsave as sp_imsave
from skimage.io import imread, imsave
from skimage import img_as_int

import os
from io import BytesIO
from base64 import b64encode
import datetime


def format_dict(d):
    s = ['']
    def helper(d, s, depth=0):
        for k,v in sorted(d.items(), key=lambda x: x[0]):
            if isinstance(v, dict):
                s[0] += ("  ")*depth + ("%s: {" % k) + ',\n'
                helper(v, s, depth+1)
                s[0] += ("  ")*depth + ("}") + ',\n'
            else:
                s[0] += ("  ")*depth + "%s: %s" % (k, v) + ',\n'
    helper(d, s)
    return s[0]


class HTMLReport:
    def __init__(self, path, images_per_row=2, default_image_width=400):
        self.path = path
        title = datetime.datetime.today().strftime(
            "Report %Y-%m-%d_%H-%M-%S_{}".format(os.uname()[1])
        )
        self.doc = dominate.document(title=title)
        self.images_per_row = images_per_row
        self.default_image_width = default_image_width
        self.t = None
        self.row_image_count = 0

    def add_header(self, str):
        with self.doc:
            h3(str, style='word-wrap: break-word; white-space: pre-wrap;')
        self.t = None
        self.row_image_count = 0
        
    def add_text(self, str):
        with self.doc:
            p(str, style='word-wrap: break-word; white-space: pre-wrap;')
        self.t = None
        self.row_image_count = 0

    def _add_table(self, border=1):
        self.row_image_count = 0
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def _encode_image(self, img_arr):
        """Save the image array as PNG and then encode with base64 for embedding"""
        img_arr = img_as_int(img_arr)
        sio = BytesIO()
        sp_imsave(sio, img_arr, 'png')
        encoded = b64encode(sio.getvalue()).decode()
        sio.close()
        return encoded

    def add_image(self, im, txt='', width=None, font_pct=100):
        if width is None:
            width = self.default_image_width
        if self.t is None or self.row_image_count >= self.images_per_row:
            self._add_table()
        with self.t:
            # with tr():
            #with td(style="word-wrap: break-word;", halign="center", valign="top"):
            with td(halign="center", valign="top"):
                with p():
                    img(
                        style="width:%dpx" % width,
                        src=r'data:image/png;base64,' + self._encode_image(im)
                    )
                    br()
                    p(
                        txt,
                        style='width:{}px; word-wrap: break-word; white-space: pre-wrap; font-size: {}%;'.format(
                            width,
                            font_pct
                        )
                    )
        self.row_image_count += 1

    def new_row(self):
        self.save()
        self.t = None
        self.row_image_count = 0

    def add_images(self, ims, txts, width=256):
        for im, txt in zip(ims, txts):
            self.add_image(im, txt, width)

    def save(self):
        f = open(self.path, 'w')
        f.write(self.doc.render())
        f.close()
        
    def __del__(self):
        self.save()

