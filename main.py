from app.cli.main import start_cli


def run() -> None:
    try:
        from app.cli.textual_app import run_textual_cli
    except ImportError:
        print("Textual is unavailable, falling back to the legacy CLI. Install textual to use the TUI.")
        start_cli()
        return

    run_textual_cli()


if __name__ == "__main__":
    run()