import dominate
import numpy as np
from dominate.tags import (a, attr, br, div, h3, h4, img, input_, meta, p,
                           source, span, table, tbody, td, th, thead, tr,
                           track, video)
from dominate.util import raw, text


class HTMLBrowser:
    def __init__(
        self,
        web_dir,
        title,
        header_template_path=None,
        filename="index.html",
        refresh=False,
    ):
        self.title = title
        self.filename = filename
        self.web_dir = web_dir
        self.filename = self.filename
        # self.web_dir.mkdir(exist_ok=True, parents=True)
        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                if header_template_path:
                    with open(header_template_path, "r") as f:
                        header_content = f.read()
                    raw(header_content)
                else:
                    meta(http_equiv="refresh", content=str(refresh))
        # wrap the main body in a skeleton "section" div
        self.body = div(cls="container")
        self.doc.add(self.body)
        self.table = None

    def add_title(self, title_text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.body:
            h4(title_text)

    def add_text(self, text_str):
        with self.body:
            raw(f"{text_str}<br>")

    def add_text_to_new_section(self, text_strs):
        with self.body:
            with div(cls="twelve columns"):
                for text_str in text_strs:
                    raw(f"{text_str}<br>")

    def add_stats(self, keys, links, probs, refs, thresholds, header=None):
        with self.body:
            with div(cls="three columns"):
                if header:
                    h4(header)
                with table():
                    with thead():
                        with tr():
                            th("Keyword")
                            for thresh in thresholds:
                                th(f">{thresh:.1f}")
                            th("all")
                            th("signbank")
                    with tbody():
                        for key, link, problist, ref in zip(keys, links, probs, refs):
                            with tr():
                                problist = np.array(problist)
                                td(a(key, href=link))
                                for thresh in thresholds:
                                    td((problist >= thresh).sum())
                                td(problist.size)
                                if ref:
                                    raw("<td>&#10003;</td>")
                                else:
                                    td()

    def add_videos(
        self, vids, txts, links, subs=None, width=320, cols_per_row=8, loop=1
    ):
        """add images to the HTML file
        Parameters:
            vids (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image,
            it will redirect you to a new page
        """
        # self.table = table(border=1, style="table-layout: fixed;")
        vid_table = table(border=0.5, cls="u-full-width")
        colors = ["black", "gold", "blue", "red", "salman"]

        # we mimic the HTML structure used for debuggin
        player_div = div(cls="player u-full-width")
        media_player_div = div(cls="mediaplayer u-full-width")

        if not subs:
            subs = [False] * len(vids)
        current_row = tr()
        tb = tbody()
        count = 0
        for (vid, txt, link, subtitle_link) in zip(vids, txts, links, subs):
            tdata = td(halign="center", valign="top")
            para = p()
            # if not (self.web_dir / vid.split("#t")[0]).exists():
            #     print(f"{vid} is missing, removing from webpage")
            #     continue

            vid_path = str(vid)
            if "#t=" in vid_path:
                offset = vid_path.split("#t=")[1].split(",")[0]
            else:
                offset = "0"
            with para:
                with a(href=str(link)):
                    with video(offset=offset):
                        kwargs = {
                            "preload": "auto",
                            "controls": "controls",
                            "autoplay": "autoplay",
                            "width": f"{width}px",
                        }
                        if loop:
                            kwargs["loop"] = "loop"
                        attr(**kwargs)
                        source(src=vid_path, type="video/mp4")
                        if subtitle_link:
                            track(
                                src=subtitle_link,
                                kind="subtitles",
                                label="English",
                                srclang="en",
                                default="default",
                            )
                br()
                text_rows = txt.split("<br>")
                for color_idx, row in enumerate(text_rows):
                    color = colors[color_idx % len(colors)]
                    bold_tag = "<b>"
                    if not row.startswith(bold_tag):
                        s_style = f"color:{color};"
                    else:
                        s_style = "color:black; font-weight: bold;"
                        row = row[len(bold_tag) :]
                    span(row, style=s_style)
            count += 1
            tdata.add(para)
            current_row.add(tdata)
            if count % cols_per_row == 0:
                tb.add(current_row)
                current_row = tr()

        # clean up trailling rows
        if current_row.children:
            tb.add(current_row)

        vid_table.add(tb)
        media_player_div.add(vid_table)
        player_div.add(media_player_div)
        # add controller div
        with player_div:
            with table():
                with tr():
                    th("Time")
                    with td():
                        input_(
                            value="0",
                            step="any",
                            type="range",
                            cls="time-slider",
                            disabled="disabled",
                        )
                with tr():
                    with td():
                        input_(
                            value="Play", type="button", cls="play",
                        )
                        input_(
                            value="Stop", type="button", cls="pause",
                        )
                        if any(subs):
                            input_(
                                value="subs", type="button", cls="subs",
                            )
        self.body.add(player_div)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = f"{self.web_dir}/{self.filename}"
        with open(html_file, "wt") as f:
            f.write(self.doc.render())