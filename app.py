import gradio as gr
import spaces

from main import (
    get_face_embeddings,
    search_photos
)


# --------------------------------------------------
# BUILD REFERENCE PEOPLE
# --------------------------------------------------

def create_reference_people(
    person_name,
    reference_files
):
    """
    Convert uploaded reference images into
    the structure expected by our recognition code.
    """

    if not person_name:
        return {}, "Please enter a person name."

    if not reference_files:
        return {}, "Please upload at least one reference photo."

    people = {
        person_name: []
    }

    for file in reference_files:

        embeddings = get_face_embeddings(file.name)

        if len(embeddings) == 0:
            continue

        # For a reference photo we expect one face.
        people[person_name].append(
            embeddings[0]
        )

    if len(people[person_name]) == 0:
        return {}, "No face was detected in the reference photos."

    message = (
        f"Loaded {len(people[person_name])} "
        f"reference photo(s) for {person_name}."
    )

    return people, message


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

@spaces.GPU
def run_search(
    person_name,
    reference_files,
    gallery_files
):

    # Validate inputs
    if not person_name:
        return [], "Please enter a person name."

    if not reference_files:
        return [], "Please upload reference photos."

    if not gallery_files:
        return [], "Please upload gallery photos."

    # Create reference embeddings
    people, message = create_reference_people(
        person_name,
        reference_files
    )

    if not people:
        return [], message

    # Search gallery
    results = search_photos(
        people,
        [file.name for file in gallery_files]
    )

    # Get matching image paths
    matching_images = []

    for result in results:

        matching_images.append(
            result["image"]
        )

    # Create result message
    if not results:

        result_message = (
            f"No photos matched {person_name}."
        )

    else:

        result_message = (
            f"Found {len(results)} matching photo(s) "
            f"for {person_name}."
        )

    return matching_images, result_message


# --------------------------------------------------
# UI
# --------------------------------------------------

with gr.Blocks(
    title="Face Search AI"
) as demo:

    gr.Markdown(
        """
        # 🔍 Face Search AI

        Upload reference photos of a person and
        search a gallery for photos containing them.
        """
    )

    # ----------------------------------------------
    # PERSON
    # ----------------------------------------------

    gr.Markdown("## 👤 Person")

    person_name = gr.Textbox(
        label="Person name",
        placeholder="Example: Tobby"
    )

    reference_files = gr.File(
        label="Reference photos",
        file_count="multiple",
        file_types=[
            ".jpg",
            ".jpeg",
            ".png"
        ]
    )

    # ----------------------------------------------
    # GALLERY
    # ----------------------------------------------

    gr.Markdown("## 📸 Gallery")

    gallery_files = gr.File(
        label="Gallery photos",
        file_count="multiple",
        file_types=[
            ".jpg",
            ".jpeg",
            ".png"
        ]
    )

    # ----------------------------------------------
    # SEARCH BUTTON
    # ----------------------------------------------

    search_button = gr.Button(
        "🔍 Search Photos"
    )

    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    result_message = gr.Textbox(
        label="Search Result"
    )

    result_gallery = gr.Gallery(
        label="Matching Photos",
        columns=3,
        height="auto"
    )

    # ----------------------------------------------
    # BUTTON ACTION
    # ----------------------------------------------

    search_button.click(
        fn=run_search,
        inputs=[
            person_name,
            reference_files,
            gallery_files
        ],
        outputs=[
            result_gallery,
            result_message
        ]
    )


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

demo.launch()
