import os
from pathlib import Path

from dotenv import load_dotenv
from fasthtml import common as fh

load_dotenv(Path(__file__).parent.parent / ".env")

tlink = fh.Script(src="https://cdn.tailwindcss.com")
hjs = fh.HighlightJS(langs=["python", "javascript", "html", "css"])
fasthtml_app, rt = fh.fast_app(
    ws_hdr=True, hdrs=[tlink, hjs], live=os.getenv("LIVE", False), debug=os.getenv("DEBUG", False)
)
app_name = "Modeldemo"
fh.setup_toasts(fasthtml_app)
root_path = "/frontend"


# Components
def github_icon():
    return fh.A(
        fh.Img(
            src=f"{root_path}/assets/gh.svg",
            alt="PyPI",
            width="50",
            height="50",
            viewBox="0 0 15 15",
            fill="none",
            cls="rounded bg-zinc-700 hover:bg-zinc-500",
        ),
        href="https://github.com/andrewhinh/modeldemo",
        target="_blank",
    )


def pypi_icon():
    return fh.A(
        fh.Img(
            src=f"{root_path}/assets/pypi.svg",
            alt="PyPI",
            width="50",
            height="50",
            viewBox="0 0 15 15",
            fill="none",
            cls="rounded bg-zinc-700 hover:bg-zinc-500",
        ),
        href="https://pypi.org/project/modeldemo/",
        target="_blank",
    )


# Layout
def main_content():
    return fh.Div(
        fh.H1("Modeldemo", cls="text-6xl font-bold text-blue-300"),
        fh.P("Stay focused.", cls="text-xl text-red-500"),
        fh.Button(
            "uv add modeldemo",
            onclick="navigator.clipboard.writeText(this.innerText);",
            hx_post="/toast",
            hx_target="#toast-container",
            hx_swap="outerHTML",
            cls="text-blue-300 p-4 rounded text-md hover:bg-zinc-700 hover:text-blue-100 cursor-pointer",
            title="Click to copy",
        ),
        fh.Div(
            github_icon(),
            pypi_icon(),
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
    fh.serve(app=app_name)


## Modal
from modal import App, Image, asgi_app

image = Image.debian_slim(python_version="3.12").pip_install("python-fasthtml")
app = App(app_name)


@app.function(image=image)
@asgi_app()
def modal_get():
    return fasthtml_app
