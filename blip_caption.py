import click
import json
import PIL
from transformers import pipeline


@click.command()
@click.argument(
    "paths",
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    nargs=-1,
    required=True,
)
@click.option("--large", is_flag=True, help="Use the large model")
@click.option("json_", "--json", is_flag=True, help="Output as JSON")
def cli(paths, large, json_):
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
        if not large
        else "Salesforce/blip-image-captioning-large",
    )
    multi = len(paths) > 1
    is_first = True
    for path, is_last in zip(paths, [False] * (len(paths) - 1) + [True]):
        if multi and not json_:
            click.echo(path)
        prefix = ""
        if json_ and is_first:
            prefix = "["
        else:
            prefix = " "
        is_first = False
        try:
            caption = captioner(str(path), max_new_tokens=100)
        except PIL.UnidentifiedImageError as ex:
            if not json_:
                click.echo(f"Error: {ex}")
            else:
                click.echo(
                    prefix
                    + json.dumps({"path": path, "error": str(ex)})
                    + ("," if not is_last else "]")
                )
            continue
        if json_:
            click.echo(
                prefix
                + json.dumps({"path": path, "caption": caption[0]["generated_text"]})
                + ("," if not is_last else "]")
            )
        else:
            click.echo(caption[0]["generated_text"])


if __name__ == "__main__":
    cli()
