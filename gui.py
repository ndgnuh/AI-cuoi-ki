from dataclasses import dataclass
import cv2
import os
import numpy as np
import typing
import functools as fp
import gi
import torch
import api
gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf
from gi.repository import Gtk


def VBox(args):
    args = [a for a in args if a is not None]
    grid = Gtk.Grid()
    for i, arg in enumerate(args):
        grid.attach(arg, 0, i, 1, 1)
    return grid


def HBox(args):
    args = [a for a in args if a is not None]
    grid = Gtk.Grid()
    for i, arg in enumerate(args):
        grid.attach(arg, i, 0, 1, 1)
    return grid


@dataclass
class MainModel():
    onchange: typing.Callable
    model_path: typing.Optional[str] = None
    image_path: typing.Optional[str] = None
    result: typing.Optional[np.array] = None

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
        if 'onchange' in self.__dict__:
            self.onchange(self)

    def model_name(self):
        if self.model_path is None:
            return "No model"
        else:
            return os.path.basename(self.model_path)

    # This is used to create quick callbacks
    # Because Python don't think anonymous function is a thing
    # Having to name the callbacks is annoying
    def set(self, attr):
        def f(v):
            if v is not None:
                setattr(self, attr, v)
        return f


def view_file_picker(parent, callback, filter=None):
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


@fp.singledispatch
def view_image(path):
    if path is None:
        return None
    else:
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, width=256, height=256, preserve_aspect_ratio=True)
        return Gtk.Image.new_from_pixbuf(pixbuf)


@view_image.register(np.ndarray)
def _(z):
    if z is None:
        return None
    z = z.astype('uint8')
    h, w, c = z.shape
    assert c == 3 or c == 4
    pixbuf = GdkPixbuf.Pixbuf.new_from_data(
        z.tobytes(),
        GdkPixbuf.Colorspace.RGB, c == 4, 8, w, h, w*c, None, None)
    return Gtk.Image.new_from_pixbuf(pixbuf)


def view(model):
    btn_sel_image = Gtk.Button.new_with_label("Select image")
    btn_sel_model = Gtk.Button.new_with_label("Select model")
    btn_run_model = Gtk.Button.new_with_label("Segment")
    # btn_sav_image = Gtk.Button.new_with_label("Save result")
    btn_reset = Gtk.Button.new_with_label("Reset")
    txt_model_name = Gtk.Label(label=model.model_name())
    image_input = view_image(model.image_path)
    image_result = view_image(model.result)

    btn_run_model.set_sensitive(
            model.model_path is not None and model.image_path is not None)

    btn_sel_image.connect("clicked", fp.partial(
        view_file_picker,
        callback=model.set("image_path")))

    btn_sel_model.connect("clicked", fp.partial(
        view_file_picker,
        callback=model.set("model_path")))

    def run_segment(_):
        if image_input is None:
            return
        image = cv2.imread(model.image_path)
        m = torch.load(model.model_path, map_location="cpu")
        result = api.remove_background(m, image)
        w = image_input.get_pixbuf().get_width()
        h = image_input.get_pixbuf().get_height()
        model.result = cv2.resize(result, (w, h))
    btn_run_model.connect("clicked", run_segment)

    def reset(_):
        model.model_path = None
        model.image_path = None
    btn_reset.connect("clicked", reset)

    return HBox([
        VBox([
            btn_sel_image,
            btn_sel_model,
            btn_run_model,
            # btn_sav_image
            btn_reset,
        ]),
        VBox([
            HBox([
                Gtk.Label(label="Selected model: "),
                txt_model_name
            ]),
            HBox([
                image_input,
                image_result
            ])
        ])
    ])


def main():
    win = Gtk.Window(title="Segmentation", role="toolbox")

    def render(model):
        for c in win.get_children():
            win.remove(c)
        win.add(view(model))
        win.show_all()

    MainModel(onchange=render)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
