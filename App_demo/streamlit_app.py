import streamlit as st
from PIL import Image
from utils import icon
from streamlit_image_select import image_select
from streamlit_drawable_canvas import st_canvas
import random
import numpy as np

import sys
sys.path.append('..')
from base_gen import load_model_base, gen_base
from image_condition_gen import load_controlnet_model, gen_controlnet
from inpainting_gen import load_model_inpaint, inpaint_gen



#pipeline
pipe = None

# UI configurations
st.set_page_config(page_title="Interor and Architect Generator",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Interior and Architerct Generation]")


# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()




def main():
    
    st.sidebar.info("**Start here ↓**", icon="👋🏾")

    option_function = st.sidebar.selectbox(
        "Choose a function", ["Generate", "Fix Style", "Replace object"], index=0)

    #take image control
    if option_function == "Fix Style":
        image_controlnet = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if image_controlnet is not None:
            image_controlnet = Image.open(image_controlnet)
            h, w = image_controlnet.size
            st.image(image_controlnet, caption='Uploaded Image', use_column_width=0.8)
            # image_controlnet.save("controlnet.jpg")
        
    elif option_function == "Replace object":
        image_inpainting = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        # print(image_inpainting)
        
        if image_inpainting is not None:
            image_inpainting = Image.open(image_inpainting)
            w, h = image_inpainting.size
            
            fill_color = "rgba(255, 255, 255, 0.0)"
            stroke_width = st.number_input("Brush Size",
                                        value=40,
                                        min_value=1,
                                        max_value=100)
            stroke_color = "rgba(255, 255, 255, 1.0)"
            bg_color = "rgba(0, 0, 0, 1.0)"
            drawing_mode = "freedraw"
            
            st.caption(
                "Draw a mask to repalce object, then click the 'Send to Streamlit' button (bottom left, with an arrow on it).")
            canvas_result = st_canvas(
                fill_color=fill_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=image_inpainting,
                update_streamlit=False,
                height=h,
                width=w,
                drawing_mode=drawing_mode,
                key="canvas",
            )
            if canvas_result.json_data is not None:
                mask = canvas_result.image_data
                mask = mask[:, :, -1] > 0
                if mask.sum() > 0:
                    mask = mask.astype(bool)
                    mask = Image.fromarray(mask.astype('uint8') * 255).convert('L')
                    mask.save("j.jpg")
                      
    
    # Form for user input
    with st.sidebar:
        with st.form("generation-form"):
            prompt = st.text_area(
                ":orange[**Enter prompt: ✍🏾**]",
                value="a modern living room with a sofa and a TV",
                height=150)
            negative_prompt = st.text_area(":orange[**Negative prompt 🙅🏽‍♂️**]",
                                            value="",
                                            help="This is a negative prompt, basically type what you don't want to see in the generated image",
                                            height=100)
            
            st.divider()
            num_outputs = st.slider(
                "Number of images to output", value=1, min_value=1, max_value=4)
            width = st.number_input("Width of output image", value=512)
            height = st.number_input("Height of output image", value=512)

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)
    
    if submitted:
        with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            
        # try: 
        if submitted:
            # Calling the replicate API to get the image
            with generated_images_placeholder.container():
                all_images = []  # List to store all generated images
                # prompt = translate_to_eng(prompt)
                # negative_prompt = translate_to_eng(negative_prompt)
                seed = random.randint(0, 100000)
                
                if option_function == "Generate":
                    pipe = load_model_base("../checkpoints/Interior.safetensors")
                    pipe.load_lora_weights("../checkpoints", weight_name="Interior_lora.safetensors")
                    pipe.fuse_lora(lora_scale=0.7)
                    output = gen_base(pipe, 
                                    prompt, 
                                    trigger_words="", 
                                    neg=negative_prompt, 
                                    num_images=num_outputs, 
                                    height=height, width=width,
                                    seed = seed)
                elif option_function == "Fix Style":
                    pipe = load_controlnet_model("../checkpoints/Interior.safetensors")
                    pipe.load_lora_weights("../checkpoints", weight_name="Interior_lora.safetensors")
                    pipe.fuse_lora(lora_scale=0.7)
                    output = gen_controlnet(pipe, 
                                        prompt, trigger_words="", 
                                        neg=negative_prompt, 
                                        num_images=num_outputs, 
                                        height=height, width=width, 
                                        image=image_controlnet,
                                        seed = seed)
                else: 
                    image_inpainting = Image.fromarray(np.array(image_inpainting))
                    mask = mask.convert("RGB")
                    
                    pipe = load_model_inpaint("../checkpoints/Interior.safetensors")
                    output = inpaint_gen(pipe,
                                        image_inpainting,
                                        mask,
                                        prompt,
                                        neg=negative_prompt, 
                                        num_images=num_outputs, 
                                        height=height, width=width,
                                        seed = seed)
                
                if output:
                    st.toast('Your image has been generated!', icon='😍')
                    # Save generated image to session state
                    st.session_state.generated_image = output

                    # Displaying the image
                    for image in st.session_state.generated_image:
                        with st.container():
                            # Display image with specified width to make it appear smaller
                            st.image(image, caption="Generated Image 🎈", use_column_width=0.8)  # Set width as desired

                            # Add image to the list
                            all_images.append(image)

                    # Save all generated images to session state
                    st.session_state.all_images = all_images
    

if __name__ == "__main__":
    
    main()
