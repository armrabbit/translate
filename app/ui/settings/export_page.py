from PySide6 import QtWidgets
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox

class ExportPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        batch_label = MLabel(self.tr("Automatic Mode")).h4()
        batch_note = MLabel(
            self.tr(
                "Selected exports are saved to comic_translate_<timestamp> in the same directory as the input file/archive."
            )
        ).secondary()
        batch_note.setWordWrap(True)
        self.raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        self.translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        self.inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))
        upscale_label = MLabel(self.tr("Image Upscaler")).h4()
        upscale_note = MLabel(
            self.tr("Upscale exported images for cleaner output (slower and larger files).")
        ).secondary()
        upscale_note.setWordWrap(True)

        upscale_layout = QtWidgets.QHBoxLayout()
        upscale_factor_label = MLabel(self.tr("Upscale factor:"))
        self.upscale_factor_combo = MComboBox().small()
        self.upscale_factor_combo.addItem(self.tr("Off (1x)"), 1)
        self.upscale_factor_combo.addItem("2x", 2)
        self.upscale_factor_combo.addItem("4x", 4)
        self.upscale_factor_combo.setCurrentIndex(0)
        upscale_layout.addWidget(upscale_factor_label)
        upscale_layout.addWidget(self.upscale_factor_combo)
        upscale_layout.addStretch(1)

        layout.addWidget(batch_label)
        layout.addWidget(batch_note)
        layout.addWidget(self.raw_text_checkbox)
        layout.addWidget(self.translated_text_checkbox)
        layout.addWidget(self.inpainted_image_checkbox)
        layout.addSpacing(8)
        layout.addWidget(upscale_label)
        layout.addWidget(upscale_note)
        layout.addLayout(upscale_layout)

        layout.addStretch(1)
