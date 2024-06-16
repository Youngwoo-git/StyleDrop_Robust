from streamlit_image_select import image_select
from glob import glob
import streamlit as st
import os
import shutil

if 'selected_imgs' not in st.session_state:
    st.session_state.selected_imgs = []
    st.session_state.selected_caps = []
if 'original_imgs' not in st.session_state:
    st.session_state.original_imgs = []
    st.session_state.original_caps = []
    
if 'style_list' not in st.session_state:
    # st.session_state.style_list = glob("./data/image*.json") + glob("./data/oriental*.json")
    st.session_state.style_list = glob("./data/image*.json")
    st.session_state.style_list = [os.path.basename(x).split(".")[0] for x in st.session_state.style_list]
    st.session_state.style_list.sort()
    
if 'style' not in st.session_state:
    st.session_state.style = ""
    
with st.form("style_box"):
    style = st.selectbox("Choose Styles to work with:", tuple(st.session_state.style_list), placeholder="Select style ...",)
    submitted = st.form_submit_button("Select")
    if submitted:
        st.session_state.style = style
        st.session_state.original_imgs = glob("./results/{}/it_data/*.png".format(st.session_state.style))
        st.session_state.original_caps = [os.path.basename(x).split(".")[0] for x in st.session_state.original_imgs]
        st.session_state.selected_imgs = []
        st.session_state.selected_caps = []
    
# with st.form("Ori_from"):
if st.session_state.original_imgs:
    st.write("Produced Samples")
    img = image_select(
        label="Select Images for Stage II",
        images=st.session_state.original_imgs,
        captions=st.session_state.original_caps,
    )

    _, col1 = st.columns([5, 1])

    # Every form must have a submit button.
    # submitted = col1.form_submit_button("Add")
    submitted = col1.button("Add")
    if submitted:
        idx = st.session_state.original_imgs.index(img)
        st.session_state.selected_imgs.append(img)
        st.session_state.selected_caps.append(st.session_state.original_caps[idx])

        st.session_state.original_imgs.pop(idx)
        st.session_state.original_caps.pop(idx)

        st.rerun()
    # print(idx)

if st.session_state.selected_imgs:
    # with st.form("New_from"):
    st.write("Selected Samples")
    img_to_remove = image_select(
        label="Select Images to remove",
        images=st.session_state.selected_imgs,
        captions=st.session_state.selected_caps,
    )

    _, col2 = st.columns([5, 1])

    # Every form must have a submit button.
    # submitted = col2.form_submit_button("Remove")
    submitted = col2.button("Remove")
    if submitted:
        idx = st.session_state.selected_imgs.index(img_to_remove)
        st.session_state.original_imgs.append(img_to_remove)
        st.session_state.original_caps.append(st.session_state.selected_caps[idx])

        st.session_state.selected_imgs.pop(idx)
        st.session_state.selected_caps.pop(idx)

        st.rerun()
            
            
save_folder_name = st.text_input("name save directory", value="")
_, col3 = st.columns([5, 1])



with col3:
    if st.button("Save", use_container_width=True):
        save_dir = "./results/{}/{}/".format(st.session_state.style, save_folder_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
        else:
            os.mkdir(save_dir)
        
        for img_path in st.session_state.selected_imgs:
            shutil.copy(img_path, save_dir)
        # st.write(save_dir)
        # st.write(st.session_state.selected_imgs)
        st.write("Saved!")