import os
from pathlib import Path

import modal

from utils import (
    MINUTES,
    NAME,
    PYTHON_VERSION,
)

parent_path: Path = Path(__file__).parent
in_prod = os.getenv("MODAL_ENVIRONMENT", "dev") == "main"

# Modal
FE_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(  # add Python dependencies
        "python-fasthtml==0.6.10", "simpleicons==7.21.0"
    )
    .copy_local_dir(parent_path, "/root/")
)

FE_TIMEOUT = 24 * 60 * MINUTES  # max
FE_CONTAINER_IDLE_TIMEOUT = 20 * MINUTES  # max
FE_ALLOW_CONCURRENT_INPUTS = 1000  # max


APP_NAME = f"{NAME}-frontend"
app = modal.App(APP_NAME)


@app.function(
    image=FE_IMAGE,
    secrets=[modal.Secret.from_dotenv(path=parent_path, filename=".env" if in_prod else ".env.dev")],
    timeout=FE_TIMEOUT,
    container_idle_timeout=FE_CONTAINER_IDLE_TIMEOUT,
    allow_concurrent_inputs=FE_ALLOW_CONCURRENT_INPUTS,
)
@modal.asgi_app()
def modal_get():
    from fasthtml import common as fh
    from simpleicons.icons import si_github, si_pypi

    fasthtml_app, rt = fh.fast_app(
        ws_hdr=True,
        hdrs=[
            fh.Script(src="https://cdn.tailwindcss.com"),
            fh.HighlightJS(langs=["python", "javascript", "html", "css"]),
            fh.Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
        ],
        live=os.getenv("LIVE", False),
        debug=os.getenv("DEBUG", False),
    )
    fh.setup_toasts(fasthtml_app)

    # Components
    def icon(
        svg,
        width="35",
        height="35",
        viewBox="0 0 15 15",
        fill="none",
        cls="rounded p-0.5 hover:bg-zinc-700 cursor-pointer",
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
    def nav():
        return fh.Nav(
            fh.A(
                f"{NAME}",
                href="/",
                cls="text-xl text-blue-300 hover:text-blue-100 font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
            ),
            fh.Div(
                fh.A(
                    icon(si_github.svg),
                    href="https://github.com/andrewhinh/eyefocus",
                    target="_blank",
                ),
                fh.A(
                    icon(si_pypi.svg),
                    href="https://pypi.org/project/eyefocus/",
                    target="_blank",
                ),
                cls="flex gap-4",
            ),
            cls="flex justify-between p-4",
        )

    def main_content():
        return fh.Main(
            fh.P(
                "Stay focused.",
                cls="text-2xl text-red-500  font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
            ),
            fh.Button(
                "uv add eyefocus",
                onclick="navigator.clipboard.writeText(this.innerText);",
                hx_post="/toast",
                hx_target="#toast-container",
                hx_swap="outerHTML",
                cls="rounded p-4 text-blue-300 text-md border border-blue-300 hover:border-blue-100 hover:bg-zinc-700 hover:text-blue-100 cursor-pointer  font-mono font-family:Consolas, Monaco, 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Courier New'",
                title="Click to copy",
            ),
            cls="flex flex-col justify-center items-center gap-8 flex-1",
        )

    def toast_container():
        return fh.Div(id="toast-container", cls="hidden")

    def footer():
        return fh.Footer(
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
        return fh.Title(NAME), fh.Div(
            nav(),
            main_content(),
            toast_container(),
            footer(),
            cls="flex flex-col justify-between min-h-screen bg-zinc-900 w-full",
        )

    @rt("/toast")
    async def toast(session):
        fh.add_toast(session, "Copied to clipboard!", "success")
        return fh.Div(id="toast-container", cls="hidden")

    @rt("/{fname:path}.{ext:static}")
    async def static_files(fname: str, ext: str):
        static_file_path = parent_path / f"{fname}.{ext}"
        if static_file_path.exists():
            return fh.FileResponse(static_file_path)

    return fasthtml_app
