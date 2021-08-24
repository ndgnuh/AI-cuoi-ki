import torch
import cv2
import os
import api
import functools as fp
import numpy as np
from gi.repository import GdkPixbuf
from gi.repository import Gtk
import gi
gi.require_version("Gtk", "3.0")


DEFAULT_MODEL="checkpoint/SegModel11Fixed256.pth"
NoneType = type(None)
state = {"model": torch.load(DEFAULT_MODEL, map_location="cpu")
        }


def Button(label, callback):
    btn = Gtk.Button.new_with_label(label)
    btn.connect("clicked", callback)
    return btn


def VBox(args):
    grid = Gtk.Grid()
    for i, arg in enumerate(args):
        grid.attach(arg, 0, i, 1, 1)
    return grid


def HBox(args):
    grid = Gtk.Grid()
    for i, arg in enumerate(args):
        grid.attach(arg, i, 0, 1, 1)
    return grid


@fp.singledispatch
def image_view(*args):
    raise NotImplementedError


# @image_view.register(str)
# def _(path, width=256, height=256):
#     p = GdkPixbuf.Pixbuf.new_from_file_at_scale(
#         path,
#         width=width,
#         height=height,
#         preserve_aspect_ratio=True)
#     return Gtk.Image.new_from_pixbuf(p)


# @image_view.register(np.array)
# def _(z, width=256, height=256):
#     z = z.astype('uint8')
#     h, w, c = z.shape
#     assert c == 3 or c == 4
#     return GdkPixbuf.Pixbuf.new_from_data(
#         z.tobytes(),
#         GdkPixbuf.Colorspace.RGB, c == 4, 8, w, h, w*c, None, None)


def convert_pixbuf2np(p):
    " convert from GdkPixbuf to numpy array"
    w, h, c, r = (p.get_width(), p.get_height(),
                  p.get_n_channels(), p.get_rowstride())
    assert p.get_colorspace() == GdkPixbuf.Colorspace.RGB
    assert p.get_bits_per_sample() == 8
    if p.get_has_alpha():
        assert c == 4
    else:
        assert c == 3
    assert r >= w * c
    a = np.frombuffer(p.get_pixels(), dtype=np.uint8)
    if a.shape[0] == w*c*h:
        return a.reshape((h, w, c))
    else:
        b = np.zeros((h, w*c), 'uint8')
        for j in range(h):
            b[j, :] = a[r*j:r*j+w*c]
        return b.reshape((h, w, c))


def convert_np2pixbuf(z):
    " convert from numpy array to GdkPixbuf "
    z = z.astype('uint8')
    h, w, c = z.shape
    assert c == 3 or c == 4
    # if hasattr(GdkPixbuf.Pixbuf, 'new_from_bytes'):
    #     Z = GLib.Bytes.new(z.tobytes())
    #     return GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, c == 4, 8, w, h, w*c)
    return GdkPixbuf.Pixbuf.new_from_data(
        z.tobytes(),
        GdkPixbuf.Colorspace.RGB, c == 4, 8, w, h, w*c, None, None)


def pick_file(parent, callback, filter=None):
    w = Gtk.FileChooserDialog(action=Gtk.FileChooserAction.OPEN)
    w.add_buttons(Gtk.STOCK_CANCEL,
                  Gtk.ResponseType.CANCEL,
                  Gtk.STOCK_OPEN,
                  Gtk.ResponseType.OK)

    if filter is not None:
        w.add_filter(filter)

    res = w.run()
    if res == Gtk.ResponseType.OK:
        callback(w.get_filename())
    else:
        callback(None)
    w.destroy()


# WIDGETS


btn_sel_model = Gtk.Button.new_with_label("Select model")
btn_sel_image = Gtk.Button.new_with_label("Select image")
btn_run_segment = Gtk.Button.new_with_label("Segment")
btn_sav_segment = Gtk.Button.new_with_label("Save result")
txt_model_file = Gtk.Label("No model selected")
img_preview = Gtk.Image.new_from_pixbuf()
img_result = Gtk.Image.new_from_pixbuf()


@fp.singledispatch
def cb_sel_image(*args, **kwargs):
    raise NotImplementedError


@cb_sel_image.register(NoneType)
def _(none):
    # print("Behavior for None not implemented")
    pass


@cb_sel_image.register(str)
def _(path):
    try:
        p = GdkPixbuf.Pixbuf.new_from_file_at_scale(
            path,
            width=256,
            height=256,
            preserve_aspect_ratio=True)
        state["image"] = cv2.imread(path)
        img_preview.set_from_pixbuf(p)
    except Exception as e:
        pass


@fp.singledispatch
def cb_sel_model(path):
    raise NotImplementedError


@cb_sel_model.register(NoneType)
def _(_):
    pass


@cb_sel_model.register(str)
def _(path):
    try:
        state["model"] = torch.load(path, map_location="cpu")
        txt_model_file.set_label(os.path.basename(path))
    except Exception as e:
        txt_model_file.set_label(str(e))
        print(e)


# TODO: add filter
btn_sel_image.connect("clicked", fp.partial(
    pick_file,
    callback=cb_sel_image))


btn_sel_model.connect("clicked", fp.partial(
    pick_file,
    callback=cb_sel_model))


def segment_image(parent):
    if 'model' not in state or 'image' not in state:
        return
    model = state['model']
    image = state['image']
    result = api.remove_background(model, image)
    w = img_preview.get_pixbuf().get_width()
    h = img_preview.get_pixbuf().get_height()
    result = cv2.resize(result, (w, h))
    result = convert_np2pixbuf(result)
    img_result.set_from_pixbuf(result)


btn_run_segment.connect("clicked", segment_image)


win = Gtk.Window(title="Segmentation", role="toolbox")


# VIEW


win.add(HBox([
    VBox([
        btn_sel_model,
        btn_sel_image,
        btn_run_segment,
        btn_sav_segment,
    ]),
    VBox([
        txt_model_file,
        Gtk.Label(""),
        HBox([
            img_preview,
            img_result
        ])
    ])
]))


win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
