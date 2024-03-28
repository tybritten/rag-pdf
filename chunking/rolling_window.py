import re
from typing import Any

import tiktoken
from colorama import Fore, Style
from semantic_router.encoders import BaseEncoder
from semantic_router.splitters import RollingWindowSplitter

from loguru import logger
from html.parser import HTMLParser


def _tiktoken_length(text: str):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


class UnstructuredSemanticSplitter:
    def __init__(
        self,
        encoder: BaseEncoder,
        window_size: int,
        min_split_tokens: int,
        max_split_tokens: int,
    ):
        self.splitter = RollingWindowSplitter(
            encoder=encoder,
            window_size=window_size,
            min_split_tokens=min_split_tokens,
            max_split_tokens=max_split_tokens,
        )
        self.max_split_tokens = max_split_tokens

    def _is_valid_title(self, title: str) -> bool:
        # Rule 1: Title starts with a lowercase letter
        if re.match(r"^[a-z]", title):
            return False
        # Rule 2: Title has a special character (excluding :, -, and .)
        if re.search(r"[^\w\s:\-\.]", title):
            return False
        # Rule 3: Title ends with a dot
        if title.endswith("."):
            return False
        return True

    def _split_table(self, table_html: str, max_split_tokens: int) -> list[str]:
        parser = TableParser()
        parser.feed(table_html)

        # Create the full table HTML to check if it needs splitting
        full_table = (
            '<table border="1" class="dataframe">'
            + parser.title_row
            + "<tbody>"
            + "".join(parser.rows)
            + "</tbody></table>"
        )

        # If the full table is within the token limit, return it without splitting
        if _tiktoken_length(full_table) <= max_split_tokens:
            return [full_table]

        splitted_tables = []  # To store split tables
        current_chunk = []

        # If the table exceeds the token limit, split it
        for row in parser.rows:
            # Temporarily add the current row to the chunk to check size
            temp_table = (
                '<table border="1" class="dataframe">'
                + parser.title_row
                + "<tbody>"
                + "".join(current_chunk + [row])
                + "</tbody></table>"
            )
            if _tiktoken_length(temp_table) > max_split_tokens:
                if current_chunk:
                    # Finalize the current chunk if it's not empty
                    splitted_tables.append(
                        '<table border="1" class="dataframe">'
                        + parser.title_row
                        + "<tbody>"
                        + "".join(current_chunk)
                        + "</tbody></table>"
                    )
                    current_chunk = [row]  # Start a new chunk with the current row
                else:
                    # If a single row exceeds the limit,
                    # add it anyway (to handle edge cases)
                    splitted_tables.append(temp_table)
                    current_chunk = []  # Reset for the next chunk
            else:
                current_chunk.append(row)  # Add the row to the current chunk

        # Add any remaining rows as a chunk
        if current_chunk:
            splitted_tables.append(
                '<table border="1" class="dataframe">'
                + parser.title_row
                + "<tbody>"
                + "".join(current_chunk)
                + "</tbody></table>"
            )

        return splitted_tables

    def _group_elements_by_title(self, elements: list[dict[str, Any]]) -> dict:
        grouped_elements = {}
        current_title = "Untitled"  # Default title for initial text without a title

        for element in elements:
            if element.get("type") == "Title":
                potential_title = element.get("text", "Untitled")
                if self._is_valid_title(potential_title):
                    current_title = potential_title
                else:
                    continue
            if current_title not in grouped_elements:
                grouped_elements[current_title] = []
            grouped_elements[current_title].append(element)
        return grouped_elements

    def split_grouped_elements(
        self, elements: list[dict[str, Any]], splitter: RollingWindowSplitter
    ) -> list[dict[str, Any]]:
        grouped_elements = self._group_elements_by_title(elements)
        chunks_with_title = []

        def _append_chunks(
            *, title: str, content: str, chunk_index: int, metadata: dict, element_type:str = "WindowElement"
        ):
            chunks_with_title.append(
                {
                    "title": title,
                    "type": element_type,
                    "text": content,
                    "chunk_index": chunk_index,
                    "metadata": metadata,
                }
            )

        for index, (title, elements) in enumerate(grouped_elements.items()):
            if not elements:
                continue
            section_metadata = elements[0].get(
                "metadata", {}
            )  # Took first element's data
            accumulated_element_texts: list[str] = []
            chunks: list[dict[str, Any]] = []

            for element in elements:
                if not element.get("text"):
                    continue
                if element.get("type") == "Table":
                    # Process accumulated text before the table
                    if accumulated_element_texts:
                        splits = splitter(accumulated_element_texts)
                        for split in splits:
                            _append_chunks(
                                title=title,
                                content=split.content,
                                chunk_index=index,
                                metadata=section_metadata,
                            )
                        # TODO: reset after PageBreak also
                        accumulated_element_texts = (
                            []
                        )  # Start new accumulation after table

                    # Add table as a separate chunk or split it if
                    table_html = element.get("metadata", {}).get("text_as_html", "")
                    splitted_tables = self._split_table(
                        table_html, self.max_split_tokens
                    )
                    metadata = {**element.get("metadata", {})}
                    metadata.pop("text_as_html", None)
                    for table in splitted_tables:
                        if not table:
                            logger.warning("Empty table encountered")
                            continue
                        _append_chunks(
                            title=title,
                            content=table,  # TODO: This should be a summary of table
                            chunk_index=index,
                            # TODO: Think of how to pass this to LLM
                            metadata={"text_as_html": table, **metadata},
                            element_type="Table",
                        )
                else:
                    accumulated_element_texts.append(element.get("text"))

            # Process any remaining accumulated text after the last table
            # or if no table was encountered

            if accumulated_element_texts:
                splits = splitter(accumulated_element_texts)
                for split in splits:
                    _append_chunks(
                        title=title,
                        content=split.content,
                        chunk_index=index,
                        metadata=section_metadata,
                    )
            if chunks:
                chunks_with_title.extend(chunks)
        return chunks_with_title

    def __call__(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.split_grouped_elements(elements, self.splitter)




class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.title_row = ""
        self.current_row = ""
        self.rows = []
        self.capture_next_row_as_title = True

    def handle_starttag(self, tag, _attrs):
        if tag == "table":
            self.in_table = True
        elif tag == "thead":
            self.in_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.in_row = True
            self.current_row = ""
        elif tag in ["td", "th"]:
            self.in_cell = True
            self.current_row += "<" + tag + ">"

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
        elif tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr":
            self.in_row = False
            self.current_row += "</tr>"
            if self.capture_next_row_as_title:
                self.title_row = self.current_row
                self.capture_next_row_as_title = False
            else:
                self.rows.append(self.current_row)
        elif tag in ["td", "th"]:
            self.in_cell = False
            self.current_row += "</" + tag + ">"

    def handle_data(self, data):
        if self.in_cell:
            self.current_row += data
