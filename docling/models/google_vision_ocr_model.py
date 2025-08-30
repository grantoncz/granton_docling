import logging
from pathlib import Path
from typing import Iterable, Optional, Type

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    GoogleVisionOcrOptions,
)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.profiling import TimeRecorder

try:
    # Google Cloud Vision API
    from google.cloud import vision
    from google.oauth2.service_account import Credentials
except ImportError:
    raise ImportError(
        "Google Cloud Vision is not installed. Please install it via `pip install google-cloud-vision` to use this OCR engine."
    )
    
import io
import numpy as np

_log = logging.getLogger(__name__)


class GoogleVisionOcrModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: GoogleVisionOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: GoogleVisionOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi.
        self.reader = None
        self.osd_reader = None
        self.script_readers = {}

        if self.enabled:
            # Initialize the GoogleVisionOcr
            cred = Credentials.from_service_account_info(options.credentials)
            self.reader:vision.ImageAnnotatorClient = vision.ImageAnnotatorClient(credentials=cred)
            _log.debug("Initializing GoogleVisionOCR: %s", str(self.reader))
            lang = "+".join(self.options.lang)
            _log.debug("Google Vision OCR supported languages: %s", lang)

    def __del__(self):
        if self.reader is not None:
            # Finalize the GoogleVisionOcr client
            if hasattr(self.reader, "close"):
                self.reader.close()

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    assert self.reader is not None  # self.reader = vision.ImageAnnotatorClient()

                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        if ocr_rect.area() == 0:
                            continue

                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )

                        # Convert PIL image to bytes
                        buf = io.BytesIO()
                        high_res_image.save(buf, format="PNG")
                        image_bytes = buf.getvalue()

                        vision_image = vision.Image(content=image_bytes)
                        response = self.reader.document_text_detection(image=vision_image)
                        if response.error.message:
                            _log.error(
                                "Google Vision OCR error: %s (code: %d)",
                                response.error.message,
                                response.error.code,
                            )
                            continue
                        
                        if len(response.text_annotations) == 0:
                            _log.error("Google Vision OCR returned no text annotations!")
                            continue
                        
                        if len(response.full_text_annotation.pages) != 1:
                            _log.warning("Google Vision OCR returned %d pages, expected 1", len(response.full_text_annotation.pages))

                        cells = []
                        ocr_page = response.full_text_annotation.pages[0]

                        for block in ocr_page.blocks:
                            for paragraph in block.paragraphs:
                                for word in paragraph.words:
                                    word_text = ''.join([symbol.text for symbol in word.symbols])
                                    conf:float = float(np.mean([symbol.confidence for symbol in word.symbols]) if word.symbols else 100.0)

                                    vertices = word.bounding_box.vertices
                                    x_vals = [v.x for v in vertices]
                                    y_vals = [v.y for v in vertices]

                                    left = min(x_vals) / self.scale
                                    top = min(y_vals) / self.scale
                                    right = max(x_vals) / self.scale
                                    bottom = max(y_vals) / self.scale

                                    cells.append(
                                        TextCell(
                                            index=len(cells),
                                            text=word_text,
                                            orig=word_text,
                                            from_ocr=True,
                                            confidence=conf,
                                            rect=BoundingRectangle.from_bounding_box(
                                                BoundingBox.from_tuple(
                                                    coord=(left, top, right, bottom),
                                                    origin=CoordOrigin.TOPLEFT,
                                                )
                                            ),
                                        )
                                    )

                        all_ocr_cells.extend(cells)

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page


    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return GoogleVisionOcrOptions
