import numpy as np
import tempfile
from PySide6.QtGui import QUndoCommand
from PySide6.QtCore import QPointF
import imkit as imk


class SetImageCommand(QUndoCommand):
    def __init__(self, parent, file_path: str, img_array: np.ndarray, 
                 display: bool = True):
        super().__init__()
        self.ct = parent
        self.file_path = file_path
        self.update_image_history(file_path, img_array)
        self.first = True
        self.display_first_time = display

    def _display_preserving_view(self, img_array: np.ndarray):
        viewer = self.ct.image_viewer
        try:
            prev_transform = viewer.transform()
            prev_scene_rect = viewer.photo.sceneBoundingRect()
            if prev_scene_rect.isNull():
                prev_scene_rect = viewer.sceneRect()
            prev_center = viewer.mapToScene(viewer.viewport().rect().center())
            rel_x = 0.5
            rel_y = 0.5
            if prev_scene_rect.width() > 0:
                rel_x = (prev_center.x() - prev_scene_rect.left()) / prev_scene_rect.width()
            if prev_scene_rect.height() > 0:
                rel_y = (prev_center.y() - prev_scene_rect.top()) / prev_scene_rect.height()
            rel_x = max(0.0, min(1.0, float(rel_x)))
            rel_y = max(0.0, min(1.0, float(rel_y)))

            viewer.display_image_array(img_array, fit=False)
            new_scene_rect = viewer.photo.sceneBoundingRect()
            if new_scene_rect.isNull():
                new_scene_rect = viewer.sceneRect()
            target_center = QPointF(
                new_scene_rect.left() + rel_x * max(0.0, new_scene_rect.width()),
                new_scene_rect.top() + rel_y * max(0.0, new_scene_rect.height()),
            )
            viewer.setTransform(prev_transform)
            viewer.centerOn(target_center)
        except Exception:
            # Fallback: never fail command execution due to viewport-state issues.
            viewer.display_image_array(img_array, fit=False)

    def redo(self):
        if self.first:
            if not self.display_first_time:
                return
            
            file_path = self.ct.image_files[self.ct.curr_img_idx]
            
            # Ensure the file has proper history initialization
            if file_path not in self.ct.current_history_index:
                self.ct.current_history_index[file_path] = 0
            if file_path not in self.ct.image_history:
                self.ct.image_history[file_path] = [file_path]
                
            current_index = self.ct.current_history_index[file_path]
            img_array = self.get_img(file_path, current_index)
            self._display_preserving_view(img_array)
            self.first = False

        if self.ct.curr_img_idx >= 0:
            file_path = self.ct.image_files[self.ct.curr_img_idx]
            
            # Ensure proper initialization
            if file_path not in self.ct.current_history_index:
                self.ct.current_history_index[file_path] = 0
            if file_path not in self.ct.image_history:
                self.ct.image_history[file_path] = [file_path]
                
            current_index = self.ct.current_history_index[file_path]
            
            if current_index < len(self.ct.image_history[file_path]) - 1:
                current_index += 1
                self.ct.current_history_index[file_path] = current_index

                img_array = self.get_img(file_path, current_index)

                self.ct.image_data[file_path] = img_array
                self._display_preserving_view(img_array)

    def undo(self):
        if self.ct.curr_img_idx >= 0:

            file_path = self.ct.image_files[self.ct.curr_img_idx]
            
            # Ensure proper initialization
            if file_path not in self.ct.current_history_index:
                self.ct.current_history_index[file_path] = 0
            if file_path not in self.ct.image_history:
                self.ct.image_history[file_path] = [file_path]
                
            current_index = self.ct.current_history_index[file_path]
            
            if current_index > 0:
                current_index -= 1
                self.ct.current_history_index[file_path] = current_index
                
                img_array = self.get_img(file_path, current_index)

                self.ct.image_data[file_path] = img_array
                self._display_preserving_view(img_array)

   
    def update_image_history(self, file_path: str, img_array: np.ndarray):
        im = self._get_current_image_for_compare(file_path)

        unchanged = False
        if im is img_array:
            unchanged = True
        elif (
            im is not None
            and hasattr(im, "shape")
            and hasattr(im, "dtype")
            and im.shape == img_array.shape
            and im.dtype == img_array.dtype
        ):
            unchanged = np.array_equal(im, img_array)

        if unchanged:
            return

        self.ct.image_data[file_path] = img_array
        
        # Update file path history
        history = self.ct.image_history[file_path]
        current_index = self.ct.current_history_index[file_path]
        
        # Remove any future history if we're not at the end
        del history[current_index + 1:]
        
        # Save new image to temp file and add to history
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=self.ct.temp_dir)
        imk.write_image(temp_file.name, img_array)
        temp_file.close()

        history.append(temp_file.name)

        # Update in-memory history if this image is loaded
        if self.ct.in_memory_history.get(file_path, []):
            in_mem_history = self.ct.in_memory_history[file_path]
            del in_mem_history[current_index + 1:]
            in_mem_history.append(img_array.copy())

        self.ct.current_history_index[file_path] = len(history) - 1

    def _get_current_image_for_compare(self, file_path: str):
        cached = self.ct.image_data.get(file_path)
        if cached is not None:
            return cached

        history = self.ct.image_history.get(file_path, [])
        if not history:
            return self.ct.load_image(file_path)

        current_index = int(self.ct.current_history_index.get(file_path, len(history) - 1))
        current_index = max(0, min(current_index, len(history) - 1))

        in_mem_history = self.ct.in_memory_history.get(file_path, [])
        if (
            in_mem_history
            and 0 <= current_index < len(in_mem_history)
            and in_mem_history[current_index] is not None
        ):
            return in_mem_history[current_index]

        hist_path = history[current_index]
        if hist_path:
            img = imk.read_image(hist_path)
            if img is not None:
                return img

        return self.ct.load_image(file_path)

    def get_img(self, file_path, current_index):
        in_mem_history = self.ct.in_memory_history.get(file_path, [])
        if (
            in_mem_history
            and 0 <= current_index < len(in_mem_history)
            and in_mem_history[current_index] is not None
        ):
            return in_mem_history[current_index]

        history = self.ct.image_history.get(file_path, [])
        if 0 <= current_index < len(history):
            img_array = imk.read_image(history[current_index])
            if img_array is not None:
                return img_array

        # Fallback: choose latest readable history frame if index is stale/corrupted.
        for hist_path in reversed(history):
            if not hist_path:
                continue
            img_array = imk.read_image(hist_path)
            if img_array is not None:
                return img_array

        # Last resort: use controller loader fallback chain.
        return self.ct.load_image(file_path)


class ToggleSkipImagesCommand(QUndoCommand):
    def __init__(self, main, file_paths: list[str], skip_status: bool):
        super().__init__()
        self.main = main
        self.file_paths = file_paths
        self.new_status = skip_status
        self.old_status = {
            path: main.image_states.get(path, {}).get('skip', False)
            for path in file_paths
        }

    def _apply_status(self, file_path: str, skip_status: bool):
        if file_path not in self.main.image_states:
            return
        self.main.image_states[file_path]['skip'] = skip_status

        try:
            idx = self.main.image_files.index(file_path)
        except ValueError:
            return

        item = self.main.page_list.item(idx)
        if item:
            fnt = item.font()
            fnt.setStrikeOut(skip_status)
            item.setFont(fnt)

    def redo(self):
        for file_path in self.file_paths:
            self._apply_status(file_path, self.new_status)

    def undo(self):
        for file_path in self.file_paths:
            self._apply_status(file_path, self.old_status.get(file_path, False))
