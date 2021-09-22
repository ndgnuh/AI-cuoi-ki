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


IMG_WIDTH = 256
IMG_HEIGHT = 256


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
    result_orig: typing.Optional[np.array] = None

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


def move_to_if_exists(filepicker, path):
    if os.path.isdir(path):
        filepicker.set_current_folder(path)


def update_image_preview(dialog):
    path = dialog.get_preview_filename()
    try:
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
    except Exception:
        dialog.set_preview_widget_active(False)
    else:
        # scale the image
        maxwidth, maxheight = IMG_WIDTH, IMG_HEIGHT
        width, height = pixbuf.get_width(), pixbuf.get_height()
        scale = min(maxwidth/width, maxheight/height)
        if scale < 1:
            width, height = int(width*scale), int(height*scale)
            pixbuf = pixbuf.scale_simple(
                    width, height,
                    GdkPixbuf.InterpType.BILINEAR)
        dialog.preview_image.set_from_pixbuf(pixbuf)
        dialog.set_preview_widget_active(True)


def pick_image(button, callback):
    filter = Gtk.FileFilter()
    Gtk.FileFilter.add_pixbuf_formats(filter)
    w = Gtk.FileChooserDialog(
            action=Gtk.FileChooserAction.OPEN,
            buttons=(
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN, Gtk.ResponseType.OK),
            filter=filter)
    w.preview_image = Gtk.Image()
    w.set_preview_widget(w.preview_image)
    w.connect("update-preview", update_image_preview)

    move_to_if_exists(w, "images")
    move_to_if_exists(w, "image")
    move_to_if_exists(w, "test_image")
    result = w.run()

    if result == Gtk.ResponseType.OK:
        fn = w.get_filename()
        callback(fn)
    w.destroy()


def pick_model(button, callback):
    f = Gtk.FileFilter()
    Gtk.FileFilter.add_pattern(f, r"*.pth")

    w = Gtk.FileChooserDialog(
            action=Gtk.FileChooserAction.OPEN,
            buttons=(
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN, Gtk.ResponseType.OK),
            filter=f)

    move_to_if_exists(w, "checkpoints")
    move_to_if_exists(w, "checkpoint")
    move_to_if_exists(w, "models")
    move_to_if_exists(w, "model")

    res = w.run()
    if res == Gtk.ResponseType.OK:
        fn = w.get_filename()
        try:
            torch.load(fn, map_location="cpu")
            callback(fn)
        except Exception as e:
            warn(str(e))
    w.destroy()


@fp.singledispatch
def view_image(path):
    if path is None:
        return None
    else:
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    path,
                    width=IMG_WIDTH,
                    height=IMG_HEIGHT,
                    preserve_aspect_ratio=True)
        except Exception as e:
            print("ERROR", e)
            return e
        return Gtk.Image.new_from_pixbuf(pixbuf)


def load_image(path, sink=None, **kwargs):
    kwargs["width"] = IMG_WIDTH
    kwargs["height"] = IMG_HEIGHT
    kwargs["preserve_aspect_ratio"] = True
    try:
        if path is None:
            return VBox([])
        if "width" in kwargs:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, **kwargs)
        else:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(path, **kwargs)
        if sink is None:
            return pixbuf
        else:
            return sink(pixbuf)
    except Exception as e:
        warn(str(e))
        return VBox([])


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


def warn(msg: str):
    d = Gtk.MessageDialog(
            parent=None,
            title="Error",
            type=Gtk.MessageType.ERROR,
            text=msg)
    d.add_buttons(Gtk.STOCK_OK, Gtk.ResponseType.OK)
    d.run()
    d.destroy()


def save(button, image, inputpath):
    f = Gtk.FileFilter()
    f.add_pixbuf_formats()

    w = Gtk.FileChooserDialog(
            action=Gtk.FileChooserAction.SAVE,
            buttons=(
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT,
                ),
            filter=f)
    w.preview_image = Gtk.Image()
    w.set_preview_widget(w.preview_image)
    w.connect("update-preview", update_image_preview)

    w.set_current_folder(os.path.dirname(inputpath))
    outfile = os.path.splitext(os.path.basename(inputpath))
    outfile = outfile[0] + "-segmented" + outfile[1]
    w.set_current_name(outfile)

    result = w.run()
    if result == Gtk.ResponseType.ACCEPT:
        try:
            fn = w.get_filename()
            # if isinstance(image, np.array):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(fn, image)
            # if isinstance(image, GdkPixbuf.Pixbuf):
            #     pixbuf = image.get_pixbuf()
            #     pixbuf.savev(fn, ext, [], [])
        except Exception as e:
            warn(str(e))
    w.destroy()


def view(win, model):
    btn_sel_image = Gtk.Button.new_with_label("Select image")
    btn_sel_model = Gtk.Button.new_with_label("Select model")
    btn_run_model = Gtk.Button.new_with_label("Segment")
    btn_exit = Gtk.Button.new_with_label("Exit")
    btn_save_image = Gtk.Button.new_with_label("Save result")
    btn_reset = Gtk.Button.new_with_label("Reset")

    txt_model_name = Gtk.Label(label=model.model_name())
    image_input = load_image(model.image_path, Gtk.Image.new_from_pixbuf)
    image_result = view_image(model.result)

    btn_run_model.set_sensitive(
            model.model_path is not None and model.image_path is not None)
    btn_save_image.set_sensitive(
            isinstance(image_result, Gtk.Image))

    btn_sel_image.connect("clicked", fp.partial(
        pick_image,
        callback=model.set("image_path")))

    btn_sel_model.connect("clicked", fp.partial(
        pick_model,
        callback=model.set("model_path")))

    btn_save_image.connect("clicked", fp.partial(
        save,
        inputpath=model.image_path,
        image=model.result_orig))

    btn_exit.connect("clicked", lambda x: Gtk.main_quit())

    def run_segment(_):
        if image_input is None:
            return
        image = cv2.imread(model.image_path)
        m = torch.load(model.model_path, map_location="cpu")
        result = api.remove_background(m, image)
        w = image_input.get_pixbuf().get_width()
        h = image_input.get_pixbuf().get_height()
        model.result_orig = result
        model.result = cv2.resize(result, (w, h))
    btn_run_model.connect("clicked", run_segment)

    def reset(_):
        model.model_path = None
        model.image_path = None
    btn_reset.connect("clicked", reset)

    if model.image_path is None:
        image_name = "None"
    else:
        image_name = os.path.normpath(str(model.image_path)).split(os.sep)
        if len(image_name) > 3:
            image_name = os.sep.join(image_name[-3:])

    return HBox([
        VBox([
            btn_sel_model,
            btn_sel_image,
            btn_run_model,
            btn_save_image,
            btn_reset,
            btn_exit
        ]),
        VBox([
            HBox([
                Gtk.Label(label="Selected image: "),
                Gtk.Label(label=image_name),
            ]),
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
        win.add(view(win, model))
        win.show_all()

    MainModel(onchange=render, model_path="checkpoint/SegModel14.pth")
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
