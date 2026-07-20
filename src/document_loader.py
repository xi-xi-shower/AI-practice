from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from docx import Document
from PIL import Image
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".txt", ".docx", ".png"})


class DocumentLoadError(Exception):
    """Raised when a supported document cannot be parsed."""


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    text: str
    source: str
    file_type: str
    metadata: dict[str, object]


# 将文档路径转换为绝对来源路径。
def _source(path: Path) -> str:
    return str(path.resolve())


# 按页提取 PDF 中的非空文本。
def _load_pdf(path: Path) -> list[DocumentChunk]:
    try:
        reader = PdfReader(path)
        chunks: list[DocumentChunk] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            chunks.append(
                DocumentChunk(
                    text=text,
                    source=_source(path),
                    file_type="pdf",
                    metadata={"page_number": page_number},
                )
            )
        return chunks
    except Exception as exc:
        raise DocumentLoadError(f"Failed to load PDF '{path}': {exc}") from exc


# 使用 UTF-8 或 GB18030 编码读取文本文件。
def _read_markdown(path: Path) -> str:
    decode_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            decode_error = exc
        except OSError as exc:
            raise DocumentLoadError(
                f"Failed to read Markdown file '{path}': {exc}"
            ) from exc

    raise DocumentLoadError(
        f"Failed to decode Markdown file '{path}' as UTF-8 or GB18030"
    ) from decode_error


# 将 Markdown 原文加载为单个文档块。
def _load_markdown(path: Path) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text=_read_markdown(path),
            source=_source(path),
            file_type="markdown",
            metadata={},
        )
    ]


# 将 TXT 原文加载为单个文档块。
def _load_text(path: Path) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text=_read_markdown(path),
            source=_source(path),
            file_type="text",
            metadata={},
        )
    ]


_ocr_engine = None


# 延迟创建并复用 RapidOCR 引擎。
def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
        except Exception as exc:
            raise DocumentLoadError(f"Failed to initialize OCR engine: {exc}") from exc
    return _ocr_engine


# 从 PNG 图片中提取 OCR 文本和图片元数据。
def _load_png(path: Path) -> list[DocumentChunk]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        result, _ = _get_ocr_engine()(str(path))
        if not result:
            return []
        text_lines = [item[1].strip() for item in result if len(item) > 1 and item[1].strip()]
        if not text_lines:
            return []
        return [
            DocumentChunk(
                text="\n".join(text_lines),
                source=_source(path),
                file_type="png",
                metadata={
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "ocr_engine": "rapidocr-onnxruntime",
                },
            )
        ]
    except DocumentLoadError as exc:
        raise DocumentLoadError(f"Failed to load PNG '{path}': {exc}") from exc
    except Exception as exc:
        raise DocumentLoadError(f"Failed to load PNG '{path}': {exc}") from exc


# 提取 DOCX 中的非空段落和表格行。
def _load_docx(path: Path) -> list[DocumentChunk]:
    try:
        document = Document(path)
        chunks: list[DocumentChunk] = []

        for paragraph_index, paragraph in enumerate(document.paragraphs):
            if not paragraph.text.strip():
                continue
            chunks.append(
                DocumentChunk(
                    text=paragraph.text,
                    source=_source(path),
                    file_type="docx",
                    metadata={
                        "content_type": "paragraph",
                        "paragraph_index": paragraph_index,
                    },
                )
            )

        for table_index, table in enumerate(document.tables):
            for row_index, row in enumerate(table.rows):
                cell_texts = [cell.text for cell in row.cells]
                if not any(text.strip() for text in cell_texts):
                    continue
                chunks.append(
                    DocumentChunk(
                        text="\t".join(cell_texts),
                        source=_source(path),
                        file_type="docx",
                        metadata={
                            "content_type": "table_row",
                            "table_index": table_index,
                            "row_index": row_index,
                        },
                    )
                )

        return chunks
    except Exception as exc:
        raise DocumentLoadError(f"Failed to load DOCX '{path}': {exc}") from exc


_LOADERS: dict[str, Callable[[Path], list[DocumentChunk]]] = {
    ".pdf": _load_pdf,
    ".md": _load_markdown,
    ".txt": _load_text,
    ".docx": _load_docx,
    ".png": _load_png,
}


# 根据文件扩展名加载单个受支持文档。
def load_document(path: str | Path) -> list[DocumentChunk]:
    """Load one supported document and return its non-empty content chunks."""
    document_path = Path(path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document does not exist: '{document_path}'")
    if not document_path.is_file():
        raise IsADirectoryError(f"Document path is not a file: '{document_path}'")

    extension = document_path.suffix.lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported document type '{extension or '<none>'}'. "
            f"Supported types: {supported}"
        )
    return loader(document_path)


# 按稳定顺序查找目录中的受支持文档。
def _document_paths(directory: Path) -> Iterable[Path]:
    paths = (
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return sorted(paths, key=lambda path: path.as_posix().casefold())


# 递归加载目录中的全部受支持文档。
def load_directory(path: str | Path) -> list[DocumentChunk]:
    """Recursively load all supported documents in a directory."""
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: '{directory}'")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: '{directory}'")

    chunks: list[DocumentChunk] = []
    for document_path in _document_paths(directory):
        chunks.extend(load_document(document_path))
    return chunks
