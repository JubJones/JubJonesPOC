from simple_poc.utils.visualization import img_to_base64


def create_gallery_html(person_crops, selected_track_id=None):
    if not person_crops:
        return ""

    gallery_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"

    for track_id, crop in person_crops.items():
        img_base64 = img_to_base64(crop)
        is_selected = (track_id == selected_track_id)
        border_style = "border: 3px solid red;" if is_selected else "border: 1px solid #ddd;"
        btn_color = "#ff6b6b" if is_selected else "#4CAF50"
        btn_text = "Untrack" if is_selected else "Track"

        gallery_html += f"""
        <div style='text-align: center; margin-bottom: 10px;'>
            <img src='data:image/jpeg;base64,{img_base64}'
                 style='width: auto; height: 120px; {border_style}'>
            <br>
            <button
                onclick='document.getElementById("track_button_{track_id}").click()'
                style='margin-top: 5px; padding: 5px 10px;
                       background-color: {btn_color};
                       color: white; border: none; border-radius: 4px; cursor: pointer;'>
                {btn_text} ID: {track_id}
            </button>
        </div>
        """

    gallery_html += "</div>"
    return gallery_html
