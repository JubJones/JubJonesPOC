from typing import Dict, Optional
import numpy as np
from simple_poc.utils.visualization import img_to_base64


def create_gallery_html(person_crops: Dict[int, np.ndarray], selected_track_id: Optional[int] = None) -> str:
    """
    Create HTML gallery of person thumbnails with track/untrack buttons.

    Args:
        person_crops: Dictionary mapping track IDs to person image crops
        selected_track_id: Currently selected track ID, if any

    Returns:
        HTML string for the gallery with clickable buttons
    """
    if not person_crops:
        return ""

    gallery_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"

    for track_id, crop in person_crops.items():
        img_base64 = img_to_base64(crop)
        is_selected = (track_id == selected_track_id)
        border_style = "border: 3px solid red;" if is_selected else "border: 1px solid #ddd;"
        btn_color = "#ff6b6b" if is_selected else "#4CAF50"
        btn_text = "Untrack" if is_selected else "Track"

        # Ensure track_id is an integer
        track_id_int = int(track_id)

        # Create a unique container ID for each person
        container_id = f"person_container_{track_id_int}"

        gallery_html += f"""
        <div id="{container_id}" style='text-align: center; margin-bottom: 10px;'>
            <img src='data:image/jpeg;base64,{img_base64}'
                 style='width: auto; height: 120px; {border_style}'>
            <br>
            <button
                id="btn_{track_id_int}"
                onclick="(function() {{
                    // Ensure the button exists before clicking
                    const trackBtn = document.getElementById('track_button_{track_id_int}');
                    if (trackBtn) {{
                        trackBtn.click();
                    }} else {{
                        console.error('Track button {track_id_int} not found');
                    }}
                }})();"
                style='margin-top: 5px; padding: 5px 10px;
                       background-color: {btn_color};
                       color: white; border: none; border-radius: 4px; cursor: pointer;'>
                {btn_text} ID: {track_id_int}
            </button>
        </div>
        """

    gallery_html += "</div>"

    # Add a small script to ensure buttons are properly connected
    gallery_html += """
    <script>
    // Ensure this runs after the page is fully loaded
    setTimeout(function() {
        // Find all track buttons in the gallery
        const trackButtons = document.querySelectorAll('[id^="btn_"]');

        // Log available track buttons for debugging
        console.log('Found ' + trackButtons.length + ' track buttons');

        // Check if corresponding hidden buttons exist
        trackButtons.forEach(btn => {
            const id = btn.id.replace('btn_', '');
            const hiddenBtn = document.getElementById('track_button_' + id);
            if (!hiddenBtn) {
                console.error('Hidden track button for ID ' + id + ' not found');
            }
        });
    }, 500);
    </script>
    """

    return gallery_html