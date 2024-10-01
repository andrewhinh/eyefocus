from fasthtml.common import H1, A, Button, Div, Img, P, Script, fast_app, serve

tlink = Script(src="https://cdn.tailwindcss.com")
fasthtml_app, rt = fast_app(ws_hdr=True, hdrs=[tlink])
root_path = "/frontend"


def github_icon():
    return A(
        Img(
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
    return A(
        Img(
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


def main_content():
    return Div(
        H1("Modeldemo", cls="text-6xl font-bold text-blue-300"),
        P("Stay focused.", cls="text-xl text-red-500"),
        Button(
            "uv add modeldemo",
            onclick="navigator.clipboard.writeText(this.innerText)",
            cls="text-blue-300 p-4 rounded text-md hover:bg-zinc-700 hover:text-blue-100 cursor-pointer",
            title="Click to copy",
        ),
        Div(
            github_icon(),
            pypi_icon(),
            cls="flex gap-8",
        ),
        cls="flex flex-col justify-center items-center gap-8 flex-1",
    )


def footer():
    return Div(
        P("Made by", cls="text-white text-lg"),
        A(
            "Andrew Hinh",
            href="https://andrewhinh.github.io/",
            cls="text-blue-300 text-lg font-bold hover:text-blue-100",
        ),
        cls="justify-end text-right p-4",
    )


@rt("/")
async def get():
    return Div(
        main_content(),
        footer(),
        cls="flex flex-col justify-between min-h-screen bg-zinc-900 w-full",
    )


if __name__ == "__main__":
    serve(app="fasthtml_app")


# Serve on Modal
from modal import App, Image, asgi_app

image = Image.debian_slim(python_version="3.12").pip_install("python-fasthtml")
app = App("frontend")


@app.function(image=image)
@asgi_app()
def get():  # noqa: F811
    return fasthtml_app
