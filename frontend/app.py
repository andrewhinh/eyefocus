import os
from pathlib import Path

from dotenv import load_dotenv
from fasthtml import common as fh
from simpleicons.icons import si_github, si_pypi

parent_path = Path(__file__).parent.parent

load_dotenv(parent_path / ".env")

tlink = fh.Script(src="https://cdn.tailwindcss.com")
hjs = fh.HighlightJS(langs=["python", "javascript", "html", "css"])
fasthtml_app, rt = fh.fast_app(
    ws_hdr=True, hdrs=[tlink, hjs], live=os.getenv("LIVE", False), debug=os.getenv("DEBUG", False)
)
fh.setup_toasts(fasthtml_app)
root_path = parent_path / "frontend"


# Components
def icon(
    svg,
    width="50",
    height="50",
    viewBox="0 0 15 15",
    fill="none",
    cls="rounded p-0.5 border border-blue-300 hover:border-blue-100 hover:bg-zinc-700 cursor-pointer",
):
    return fh.Svg(
        fh.NotStr(svg),
        width=width,
        height=height,
        viewBox=viewBox,
        fill=fill,
        cls=cls,
    )


# Layout
def main_content():
    return fh.Div(
        fh.H1("Modeldemo", cls="text-6xl font-bold text-blue-300"),
        fh.P("Stay focused.", cls="text-2xl text-red-500"),
        fh.Button(
            "uv add modeldemo",
            onclick="navigator.clipboard.writeText(this.innerText);",
            hx_post="/toast",
            hx_target="#toast-container",
            hx_swap="outerHTML",
            cls="rounded p-4 text-blue-300 text-md border border-blue-300 hover:border-blue-100 hover:bg-zinc-700 hover:text-blue-100 cursor-pointer",
            title="Click to copy",
        ),
        fh.Div(
            fh.A(
                icon(si_github.svg),
                href="https://github.com/andrewhinh/modeldemo",
                target="_blank",
            ),
            fh.A(
                icon(si_pypi.svg),
                href="https://pypi.org/project/modeldemo/",
                target="_blank",
            ),
            cls="flex gap-8",
        ),
        cls="flex flex-col justify-center items-center gap-8 flex-1",
    )


def toast_container():
    return fh.Div(id="toast-container", cls="hidden")


def footer():
    return fh.Div(
        fh.P("Made by", cls="text-white text-lg"),
        fh.A(
            "Andrew Hinh",
            href="https://andrewhinh.github.io/",
            cls="text-blue-300 text-lg font-bold hover:text-blue-100",
        ),
        cls="justify-end text-right p-4",
    )


# Routes
@rt("/")
async def get():
    return fh.Div(
        main_content(),
        toast_container(),
        footer(),
        cls="flex flex-col justify-between min-h-screen bg-zinc-900 w-full",
    )


@rt("/toast")
async def toast(session):
    fh.add_toast(session, "Copied to clipboard!", "success")
    return fh.Div(id="toast-container", cls="hidden")


# Serving
if __name__ == "__main__":
    fh.serve(app="fasthtml_app")


## Modal
from modal import App, Image, asgi_app

image = Image.debian_slim(python_version="3.12").pip_install("python-fasthtml")
app_name = "Modeldemo"
app = App(app_name)


@app.function(image=image)
@asgi_app()
def modal_get():
    return fasthtml_app
