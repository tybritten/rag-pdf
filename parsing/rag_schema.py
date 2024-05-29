from typing import List, Optional
from enum import Enum
from torch import Tensor
from uuid import uuid4
from dataclasses import dataclass
import json


class DataType(str, Enum):
    TITLE = "Title"
    TEXT = "Text"
    UNCATEGORIZED_TEXT = "UncategorizedText"
    NARRATIVE_TEXT = "NarrativeText"
    BULLETED_TEXT = "BulletedText"
    PARAGRAPH = "Paragraph"
    ABSTRACT = "Abstract"
    THREADING = "Threading"
    FORM = "Form"
    FIELD_NAME = "Field-Name"
    VALUE = "Value"
    LINK = "Link"
    COMPOSITE_ELEMENT = "CompositeElement"
    IMAGE = "Image"
    PICTURE = "Picture"
    FIGURE_CAPTION = "FigureCaption"
    FIGURE = "Figure"
    CAPTION = "Caption"
    LIST = "List"
    LIST_ITEM = "ListItem"
    LIST_ITEM_OTHER = "List-item"
    CHECKED = "Checked"
    UNCHECKED = "Unchecked"
    CHECK_BOX_CHECKED = "CheckBoxChecked"
    CHECK_BOX_UNCHECKED = "CheckBoxUnchecked"
    RADIO_BUTTON_CHECKED = "RadioButtonChecked"
    RADIO_BUTTON_UNCHECKED = "RadioButtonUnchecked"
    ADDRESS = "Address"
    EMAIL_ADDRESS = "EmailAddress"
    PAGE_BREAK = "PageBreak"
    FORMULA = "Formula"
    TABLE = "Table"
    HEADER = "Header"
    HEADLINE = "Headline"
    SUB_HEADLINE = "Subheadline"
    PAGE_HEADER = "Page-header"  # Title?
    SECTION_HEADER = "Section-header"
    FOOTER = "Footer"
    FOOTNOTE = "Footnote"
    PAGE_FOOTER = "Page-footer"
    PAGE_NUMBER = "PageNumber"
    CODE_SNIPPET = "CodeSnippet"


class Metadata(dict):
    """Metadata fields that pertain to the data source."""
    source: str
    page_number: Optional[int] = None
    url: Optional[str] = None
    text_as_html: Optional[str] = None
    tag: Optional[str] = None


class DataElement(dict):
    """A data element is a piece of text, image, link, or table."""
    """ The content field can contain text or Base64 encoded image data."""
    id: uuid4
    data_type: DataType
    content: str | bytes
    metadata: Metadata
    embeddings: Optional[Tensor] = None



class Document(List[DataElement]):
    """A document is a list of data elements."""
    def from_dict(self, data: dict):
        for element in data:
            self.append(DataElement(
                id=element["element_id"],
                data_type=DataType(element["type"]),
                content=element["text"],
                metadata=Metadata(
                    source=element["metadata"]["source"],
                    page_number=element["metadata"]["page_number"],
                    url=element["metadata"]["url"],
                    text_as_html=element["metadata"]["text_as_html"]
                )
            ))
        return self

