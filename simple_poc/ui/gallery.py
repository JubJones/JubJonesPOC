from typing import Dict, Optional
import numpy as np
from simple_poc.utils.visualization import img_to_base64


def create_gallery_html(person_crops: Dict[str, Dict[int, np.ndarray]], selected_track_id: Optional[int] = None) -> str:
    """
    Create HTML gallery of person thumbnails with track/untrack buttons.
    Handles per-camera crops and potential track ID collisions by displaying
    a unique entry for each camera/track_id pair.

    Args:
        person_crops: Dictionary mapping camera IDs to dictionaries of track IDs and person image crops (RGB).
        selected_track_id: Currently selected track ID, if any.

    Returns:
        HTML string for the gallery with clickable buttons.
    """
    if not person_crops:
        return "<p>No people detected yet.</p>" # Provide feedback if empty

    gallery_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-end;'>" # Added gap/alignment

    # Keep track of unique track IDs displayed to handle selection highlight correctly
    # (Since selection is based on track_id only, highlight all instances of that ID)

    # Iterate through cameras and then track IDs within each camera
    all_entries = []
    for cam_id, crops_in_cam in person_crops.items():
        for track_id, crop_rgb in crops_in_cam.items():
             all_entries.append({"cam_id": cam_id, "track_id": track_id, "crop": crop_rgb})

    # Sort entries maybe by camera then track ID for consistent display order
    all_entries.sort(key=lambda x: (x['cam_id'], x['track_id']))


    for entry in all_entries:
        cam_id = entry['cam_id']
        track_id = entry['track_id']
        crop_rgb = entry['crop']

        # print(f"Processing gallery entry: Cam {cam_id}, Track {track_id}, Type: {type(crop_rgb)}") # Debug print

        if not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
            print(f"Warning: Invalid crop data for Cam {cam_id}, Track {track_id}. Skipping gallery entry.")
            continue

        try:
            # Convert RGB numpy array to base64 jpeg
            img_base64 = img_to_base64(crop_rgb) # Assumes img_to_base64 handles RGB correctly
        except Exception as e:
             print(f"Error converting crop to base64 for Cam {cam_id}, Track {track_id}: {e}")
             continue # Skip this entry if conversion fails

        # Determine if this specific track_id matches the selected one
        is_selected = (track_id == selected_track_id)
        border_style = "border: 3px solid red;" if is_selected else "border: 1px solid #ddd;"
        btn_color = "#ff6b6b" if is_selected else "#4CAF50" # Red if selected, Green otherwise
        btn_text = "Selected" if is_selected else "Select"

        # Use the original track_id for button functionality (assumes selection is global by ID)
        track_id_int = int(track_id)
        # Create a unique ID for the HTML container element if needed, though button ID is key
        # container_id = f"person_container_{cam_id}_{track_id_int}" # Example unique container ID

        # Generate unique button ID for Gradio backend to find
        # IMPORTANT: Gradio button elem_id needs to match the hidden buttons created in app.py
        gradio_button_elem_id = f"track_button_{track_id_int}"

        gallery_html += f"""
        <div style='text-align: center; margin-bottom: 10px; padding: 5px; background-color: #f8f8f8; border-radius: 5px;'>
            <img src='data:image/jpeg;base64,{img_base64}'
                 alt='Crop for ID {track_id} from Cam {cam_id}'
                 style='width: auto; height: 120px; {border_style} object-fit: contain; display: block; margin-left: auto; margin-right: auto;'>
            <div style='font-size: 0.9em; margin-top: 4px; color: #555;'>Cam: {cam_id} / ID: {track_id_int}</div>
            <button
                id="btn_display_{cam_id}_{track_id_int}" onclick="(function() {{
                    // Find the hidden Gradio button corresponding to the track_id
                    const trackBtn = document.getElementById('{gradio_button_elem_id}');
                    if (trackBtn) {{
                        console.log('Clicked gallery button for {cam_id}-{track_id_int}, triggering Gradio button: {gradio_button_elem_id}');
                        trackBtn.click(); // Trigger the hidden Gradio button
                    }} else {{
                        console.error('Gradio button {gradio_button_elem_id} not found for gallery click.');
                    }}
                }})();"
                style='margin-top: 5px; padding: 5px 10px;
                       background-color: {btn_color};
                       color: white; border: none; border-radius: 4px; cursor: pointer; width: 90%;'>
                {btn_text}
            </button>
        </div>
        """

    gallery_html += "</div>"

    # (JavaScript for debugging hidden buttons - unchanged, but less critical now)
    gallery_html += """
    <script>
    setTimeout(function() {
      const hiddenButtons = document.querySelectorAll('[id^="track_button_"]');
      console.log('Found ' + hiddenButtons.length + ' hidden Gradio track buttons on load.');
    }, 500);
    </script>
    """

    return gallery_html
