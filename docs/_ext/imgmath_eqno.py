import os
from pathlib import Path

from docutils import nodes

from sphinx.locale import _
from sphinx.ext.imgmath import (
    render_math,
    render_maths_to_base64,
    wrap_displaymath,
    get_tooltip,
    MathExtError,
)
from sphinx.util.math import get_node_equation_number


def html_visit_displaymath_right(self, node):
    config = self.builder.config

    if node.get("no-wrap", node.get("nowrap", False)):
        latex = node.astext()
    else:
        latex = wrap_displaymath(node.astext(), None, False)

    try:
        rendered_path, _depth = render_math(self, latex, config=config)
    except MathExtError as exc:
        msg = str(exc)
        sm = nodes.system_message(
            msg, type="WARNING", level=2, backrefs=[], source=node.astext()
        )
        sm.walkabout(self)
        raise nodes.SkipNode from exc

    self.body.append(self.starttag(node, "div", CLASS="math"))
    self.body.append("<p>")

    # image first
    if rendered_path is None:
        self.body.append(
            f'<span class="math">{self.encode(node.astext()).strip()}</span>'
        )
    else:
        if config.imgmath_embed:
            image_format = config.imgmath_image_format.lower()
            img_src = render_maths_to_base64(image_format, rendered_path)
        else:
            bname = os.path.basename(rendered_path)
            img_src = Path(self.builder.imgpath, "math", bname).as_posix()

        tooltip = get_tooltip(self, node, config=config)
        self.body.append(f'<img src="{img_src}"{tooltip}/>')

    # equation number after image
    if node["number"]:
        number = get_node_equation_number(self, node)
        self.body.append(f'<span class="eqno">({number})')
        self.add_permalink_ref(node, _("Link to this equation"))
        self.body.append("</span>")

    self.body.append("</p>\n</div>")

    raise nodes.SkipNode


def replace_imgmath_renderer(app):
    app.registry.html_block_math_renderers["imgmath"] = (
        html_visit_displaymath_right,
        lambda self, node: None,
    )


def setup(app):
    app.connect("builder-inited", replace_imgmath_renderer)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
