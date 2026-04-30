def import_tkinter():
    try:
        from tkinter import _flatten
    except ImportError:
        def _flatten(seq):
            def _inner(seq):
                for item in seq:
                    if isinstance(item, (list, tuple)):
                        yield from _inner(item)
                    else:
                        yield item
            return tuple(_inner(seq))