from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "project_map.md"
OUTPUT = ROOT / "project_map.pdf"

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 50
TOP = 790
LINE_HEIGHT = 14
FONT_SIZE = 10
MAX_CHARS = 86


def pdf_escape(text):
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def markdown_to_lines(markdown):
    lines = []
    in_code = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            lines.append("")
            continue

        if not line:
            lines.append("")
            continue

        if in_code:
            wrapped = textwrap.wrap(line, width=MAX_CHARS, replace_whitespace=False)
            lines.extend(wrapped or [""])
            continue

        if line.startswith("# "):
            lines.append(line[2:].upper())
            lines.append("")
            continue

        if line.startswith("## "):
            lines.append(line[3:])
            lines.append("")
            continue

        if line.startswith("### "):
            lines.append(line[4:])
            continue

        wrapped = textwrap.wrap(line, width=MAX_CHARS)
        lines.extend(wrapped or [""])

    return lines


def paginate(lines):
    pages = []
    current = []
    max_lines = int((TOP - 50) / LINE_HEIGHT)

    for line in lines:
        if len(current) >= max_lines:
            pages.append(current)
            current = []
        current.append(line)

    if current:
        pages.append(current)

    return pages


def page_stream(lines):
    commands = [
        "BT",
        f"/F1 {FONT_SIZE} Tf",
        f"{LEFT} {TOP} Td",
        f"{LINE_HEIGHT} TL",
    ]

    for line in lines:
        commands.append(f"({pdf_escape(line)}) Tj")
        commands.append("T*")

    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def write_pdf(pages):
    objects = []

    def add_object(content):
        objects.append(content)
        return len(objects)

    catalog_id = add_object("<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(None)
    font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids = []
    for page_lines in pages:
        stream = page_stream(page_lines)
        stream_id = add_object(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_id = add_object(
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {stream_id} 0 R >>"
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"

    output = bytearray()
    output.extend(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        if isinstance(obj, bytes):
            output.extend(obj)
        else:
            output.extend(obj.encode("latin-1", errors="replace"))
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    output.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )

    OUTPUT.write_bytes(output)


def main():
    markdown = SOURCE.read_text(encoding="utf-8")
    lines = markdown_to_lines(markdown)
    pages = paginate(lines)
    write_pdf(pages)
    print(f"Creato PDF: {OUTPUT}")


if __name__ == "__main__":
    main()
