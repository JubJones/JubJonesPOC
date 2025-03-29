from typing import Dict, Optional, Union  # Added Union

import numpy as np

from simple_poc.utils.visualization import img_to_base64


def create_gallery_html(person_crops: Dict[str, Dict[Union[int, str], np.ndarray]],
                        selected_track_id: Optional[int] = None) -> str:
    """
    Create HTML gallery of person thumbnails. Handles actual track IDs (int)
    and simple detection keys (str, e.g., 'det_0'). Disables selection for simple detections.

    Args:
        person_crops: Dict mapping camera IDs to dicts of (track_id/det_key) and person crops (RGB).
        selected_track_id: Currently selected integer track ID, if any.

    Returns:
        HTML string for the gallery.
    """
    if not person_crops:
        return "<p>No people detected yet.</p>"

    gallery_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-end;'>"

    all_entries = []
    for cam_id, crops_in_cam in person_crops.items():
        # Use keys() directly which gives Union[int, str]
        for storage_key in crops_in_cam.keys():
            crop_rgb = crops_in_cam[storage_key]
            all_entries.append({"cam_id": cam_id, "key": storage_key, "crop": crop_rgb})

    # Sort entries maybe by camera then key (string keys will sort after ints)
    all_entries.sort(key=lambda x: (x['cam_id'], str(x['key'])))  # Convert key to str for sorting

    for entry in all_entries:
        cam_id = entry['cam_id']
        storage_key = entry['key']  # This is Union[int, str]
        crop_rgb = entry['crop']

        if not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0: continue

        try:
            img_base64 = img_to_base64(crop_rgb)
        except Exception as e:
            print(f"Error converting crop to base64 for Cam {cam_id}, Key {storage_key}: {e}")
            continue

        is_detection = isinstance(storage_key, str) and storage_key.startswith("det_")
        is_tracked_id = isinstance(storage_key, int)

        # Determine selection highlight (only for actual integer track IDs)
        is_selected = is_tracked_id and (storage_key == selected_track_id)
        border_style = "border: 3px solid red;" if is_selected else "border: 1px solid #ddd;"

        # Determine button text and state
        button_enabled = is_tracked_id  # Only enable button for actual track IDs
        gradio_button_elem_id = None
        btn_text = "Detection"
        btn_color = "#aaaaaa"  # Grey for disabled/detection
        onclick_js = ""

        if button_enabled:
            track_id_int = int(storage_key)  # We know it's an int here
            gradio_button_elem_id = f"track_button_{track_id_int}"
            btn_color = "#ff6b6b" if is_selected else "#4CAF50"  # Red if selected, Green otherwise
            btn_text = "Selected" if is_selected else "Select"
            onclick_js = f"""
                (function() {{
                    const trackBtn = document.getElementById('{gradio_button_elem_id}');
                    if (trackBtn) {{ trackBtn.click(); }}
                    else {{ console.error('Gradio button {gradio_button_elem_id} not found.'); }}
                }})();
             """

        # Display label appropriately
        display_label = f"ID: {storage_key}" if is_tracked_id else f"Det: {storage_key.split('_')[-1]}"

        gallery_html += f"""
        <div style='text-align: center; margin-bottom: 10px; padding: 5px; background-color: #f8f8f8; border-radius: 5px;'>
            <img src='data:image/jpeg;base64,{img_base64}'
                 alt='Crop for {display_label} from Cam {cam_id}'
                 style='width: auto; height: 120px; {border_style} object-fit: contain; display: block; margin-left: auto; margin-right: auto;'>
            <div style='font-size: 0.9em; margin-top: 4px; color: #555;'>Cam: {cam_id} / {display_label}</div>
            <button
                {"disabled" if not button_enabled else ""}
                onclick="{onclick_js if button_enabled else ""}"
                style='margin-top: 5px; padding: 5px 10px;
                       background-color: {btn_color};
                       color: white; border: none; border-radius: 4px;
                       cursor: {"pointer" if button_enabled else "default"};
                       width: 90%; {"opacity: 0.6;" if not button_enabled else ""}'>
                {btn_text}
            </button>
        </div>
        """

    gallery_html += "</div>"
    return gallery_html
